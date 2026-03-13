import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone as sklearn_clone
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import xgboost as xgb
import lightgbm as lgb
import optuna
import os

# Suppress Optuna's verbose trial-by-trial logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_processed_data(filepath):
    """Load processed feature data."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath, index_col='Date', parse_dates=True)

def prepare_data(df, target_col='Target_Copper_Return_3M'):
    """Prepare data, using RETURN as target."""
    print(f"Data Shape: {df.shape}")
    
    # We need to filter out rows where target is NaN (for training)
    # But keep the ones where target is NaN (for inference)
    data_known = df.dropna(subset=[target_col])
    data_future = df[df[target_col].isna()]
    
    # Also check if features are clean (no NaNs allowed in models)
    # The models usually can't handle NaNs.
    # Note: features.py already dropped feature NaNs.
    
    print(f"Train/Test Data (Known Target): {data_known.shape}")
    print(f"Inference Data (Unknown Target): {data_future.shape}")
    
    return data_known, data_future

def train_lightgbm_model(X_train, y_train, tune=False):
    """Train LightGBM with optional Optuna Bayesian Hyperparameter Tuning."""
    if tune:
        print("Tuning LightGBM with Optuna (80 trials)...")
        tscv = TimeSeriesSplit(n_splits=5)

        def objective(trial):
            params = {
                'objective': 'regression',
                'random_state': 42,
                'verbose': -1,
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'num_leaves': trial.suggest_int('num_leaves', 8, 128),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 3, 30),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            }
            model = lgb.LGBMRegressor(**params)
            scores = cross_val_score(model, X_train, y_train, cv=tscv,
                                     scoring='neg_mean_squared_error')
            return -scores.mean()  # Optuna minimizes

        study = optuna.create_study(direction='minimize',
                                     pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=80, show_progress_bar=True)

        print(f"Best LightGBM Params: {study.best_params}")
        print(f"Best CV MSE: {study.best_value:.6f}")

        best = lgb.LGBMRegressor(
            objective='regression', random_state=42, verbose=-1,
            **study.best_params
        )
        best.fit(X_train, y_train)
        return best
    else:
        model = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)
        return model

def train_elasticnet_model(X_train, y_train):
    """Train ElasticNet with built-in CV for alpha and l1_ratio tuning.
    
    ElasticNet is a regularized linear model that combines L1 (Lasso) and L2 (Ridge)
    penalties. It adds real diversity to the ensemble since it's not tree-based.
    Pipeline includes StandardScaler since linear models are scale-sensitive.
    """
    print("Training ElasticNet (with CV)...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    # ElasticNetCV automatically tunes alpha and l1_ratio
    enet = Pipeline([
        ('scaler', StandardScaler()),
        ('enet', ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            n_alphas=50,
            cv=tscv,
            max_iter=5000,
            random_state=42
        ))
    ])
    enet.fit(X_train, y_train)
    
    best_enet = enet.named_steps['enet']
    n_nonzero = np.sum(best_enet.coef_ != 0)
    print(f"Best ElasticNet alpha: {best_enet.alpha_:.6f}, l1_ratio: {best_enet.l1_ratio_:.2f}")
    print(f"Non-zero coefficients: {n_nonzero}/{len(best_enet.coef_)}")
    
    return enet

def train_xgboost_model(X_train, y_train, tune=False):
    """Train XGBoost Model with optional Optuna Bayesian Hyperparameter Tuning."""
    if tune:
        print("Tuning XGBoost with Optuna (80 trials)...")
        tscv = TimeSeriesSplit(n_splits=5)

        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'random_state': 42,
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            }
            model = xgb.XGBRegressor(**params)
            scores = cross_val_score(model, X_train, y_train, cv=tscv,
                                     scoring='neg_mean_squared_error')
            return -scores.mean()  # Optuna minimizes

        study = optuna.create_study(direction='minimize',
                                     pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=80, show_progress_bar=True)

        print(f"Best XGB Params: {study.best_params}")
        print(f"Best CV MSE: {study.best_value:.6f}")

        best = xgb.XGBRegressor(
            objective='reg:squarederror', random_state=42,
            **study.best_params
        )
        best.fit(X_train, y_train)
        return best
    else:
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model

def evaluate_and_plot(model, X_train, y_train, X_test, y_test, prices_test, current_prices_test, model_name, momentum_correction=None, alpha_override=None):
    """
    Evaluate Model and Plot Results (Actual vs Predicted).
    momentum_correction: if provided, a DataFrame slice with momentum cols for bias correction.
    alpha_override: if provided, use this alpha instead of the default 0.3.
    """
    print(f"\n--- Evaluation: {model_name} ---")
    
    # 1. Standard Sklearn / XGBoost Prediction
    y_pred = model.predict(X_test)
    
    # 2. Apply momentum bias correction if provided
    if momentum_correction is not None:
        alpha = alpha_override if alpha_override is not None else 0.3
        y_pred = apply_momentum_correction(y_pred, momentum_correction, alpha=alpha)

    # 2. Convert Returns Back to Prices
    # Price_t+3 = Price_t * (1 + Return_3m)
    # y_pred is the predicted 3-month return
    prices_pred = current_prices_test * (1 + y_pred)

    # 3. Calculate Metrics (On Returns and Prices)
    rmse_ret = np.sqrt(mean_squared_error(y_test, y_pred))
    mae_ret = mean_absolute_error(y_test, y_pred)
    r2_ret = r2_score(y_test, y_pred)

    rmse_price = np.sqrt(mean_squared_error(prices_test, prices_pred))
    mae_price = mean_absolute_error(prices_test, prices_pred)

    print(f"Metrics (Returns):\nRMSE: {rmse_ret:.4f}, MAE: {mae_ret:.4f}, R2: {r2_ret:.4f}")
    print(f"Metrics (Prices):\nRMSE: ${rmse_price:.2f}, MAE: ${mae_price:.2f}")

    return prices_pred

def plot_feature_importance(model, feature_names, title="Feature Importance"):
    """Plot feature importance for tree-based models."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.bar(range(len(feature_names)), importances[indices], align="center")
    plt.xticks(range(len(feature_names)), feature_names[indices], rotation=90)
    plt.xlim([-1, len(feature_names)])
    plt.tight_layout()
    
    output_path = 'outputs/figures/feature_importance.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Feature importance plot saved to {output_path}")
    
    # Print top 10 features
    print("\nTop 10 Important Features:")
    for f in range(min(10, len(feature_names))):
        print(f"{f+1}. {feature_names[indices[f]]} ({importances[indices[f]]:.4f})")

def select_top_features(model, feature_names, top_n=20):
    """Select top N features based on model's feature importance."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    selected = feature_names[indices]
    
    print(f"\nSelected Top {top_n} Features:")
    for i, feat in enumerate(selected):
        print(f"  {i+1}. {feat} ({importances[indices[i]]:.4f})")
    
    return selected.tolist()

def build_stacking_ensemble(xgb_model, lgbm_model, enet_model, X_train, y_train):
    """
    Build a manual Stacking Ensemble that combines XGBoost, LightGBM, and ElasticNet.
    Uses out-of-fold predictions from TimeSeriesSplit to train a RidgeCV meta-learner.
    Three diverse base learners: two gradient boosters (different strategies) + one linear.
    """
    print("\nBuilding Stacking Ensemble (XGB + LightGBM + ElasticNet)...")
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Collect out-of-fold predictions for all 3 models
    oof_xgb = np.full(len(y_train), np.nan)
    oof_lgbm = np.full(len(y_train), np.nan)
    oof_enet = np.full(len(y_train), np.nan)
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]
        
        # Clone and fit XGBoost
        xgb_clone = xgb.XGBRegressor(**{k: v for k, v in xgb_model.get_params().items() if k != 'missing'})
        xgb_clone.fit(X_tr, y_tr)
        oof_xgb[val_idx] = xgb_clone.predict(X_val)
        
        # Clone and fit LightGBM
        lgbm_clone = lgb.LGBMRegressor(**{k: v for k, v in lgbm_model.get_params().items()})
        lgbm_clone.fit(X_tr, y_tr)
        oof_lgbm[val_idx] = lgbm_clone.predict(X_val)
        
        # Fit ElasticNet (Pipeline with scaler)
        enet_clone = Pipeline([
            ('scaler', StandardScaler()),
            ('enet', ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.9],
                n_alphas=20,
                cv=3,
                max_iter=5000,
                random_state=42
            ))
        ])
        enet_clone.fit(X_tr, y_tr)
        oof_enet[val_idx] = enet_clone.predict(X_val)
    
    # Only use rows that have valid out-of-fold predictions
    valid_mask = ~np.isnan(oof_xgb) & ~np.isnan(oof_lgbm) & ~np.isnan(oof_enet)
    meta_X = np.column_stack([oof_xgb[valid_mask], oof_lgbm[valid_mask], oof_enet[valid_mask]])
    meta_y = y_train.values[valid_mask]
    
    # Train meta-learner (RidgeCV finds optimal alpha automatically)
    meta_learner = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    meta_learner.fit(meta_X, meta_y)
    
    weights = meta_learner.coef_
    print(f"Meta-Learner Weights -> XGB: {weights[0]:.3f}, LightGBM: {weights[1]:.3f}, ElasticNet: {weights[2]:.3f}")
    print(f"Meta-Learner Intercept: {meta_learner.intercept_:.4f}")
    print(f"Meta-Learner Alpha: {meta_learner.alpha_:.4f}")
    
    # Refit base models on FULL training data for final predictions
    xgb_final = xgb.XGBRegressor(**{k: v for k, v in xgb_model.get_params().items() if k != 'missing'})
    xgb_final.fit(X_train, y_train)
    
    lgbm_final = lgb.LGBMRegressor(**{k: v for k, v in lgbm_model.get_params().items()})
    lgbm_final.fit(X_train, y_train)
    
    enet_final = Pipeline([
        ('scaler', StandardScaler()),
        ('enet', ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            n_alphas=20,
            cv=3,
            max_iter=5000,
            random_state=42
        ))
    ])
    enet_final.fit(X_train, y_train)
    
    return {
        'meta_learner': meta_learner,
        'xgb': xgb_final,
        'lgbm': lgbm_final,
        'enet': enet_final
    }

def predict_stacked(stack_dict, X):
    """Generate predictions from the stacking ensemble."""
    pred_xgb = stack_dict['xgb'].predict(X)
    pred_lgbm = stack_dict['lgbm'].predict(X)
    pred_enet = stack_dict['enet'].predict(X)
    meta_X = np.column_stack([pred_xgb, pred_lgbm, pred_enet])
    return stack_dict['meta_learner'].predict(meta_X)

def _build_correction_signal(df_slice):
    """Build the raw correction signal from momentum columns."""
    if 'Copper_Accel_1M' in df_slice.columns:
        accel = df_slice['Copper_Accel_1M'].values
    elif 'Copper_Return_1M' in df_slice.columns:
        ret1m = df_slice['Copper_Return_1M'].values
        accel = np.diff(ret1m, prepend=ret1m[0])
    else:
        return None
    
    divergence = np.zeros_like(accel)
    if 'Momentum_Divergence_1v3' in df_slice.columns:
        divergence = df_slice['Momentum_Divergence_1v3'].values
    
    return accel + 0.5 * divergence

def optimize_alpha(model, X_train, y_train, momentum_train, predict_fn=None):
    """
    Find the optimal momentum correction alpha via time-series cross-validation.
    Tests alpha from 0.0 to 0.8 and picks the one with lowest RMSE.
    """
    signal = _build_correction_signal(momentum_train)
    if signal is None:
        return 0.0
    
    tscv = TimeSeriesSplit(n_splits=5)
    best_alpha = 0.0
    best_rmse = np.inf
    
    for alpha in np.arange(0.0, 0.85, 0.05):
        fold_errors = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            if predict_fn:
                # For stacked models
                y_pred_val = predict_fn(X_val)
            else:
                # Clone and fit (works for both simple models and Pipelines)
                clone = sklearn_clone(model)
                clone.fit(X_tr, y_tr)
                y_pred_val = clone.predict(X_val)
            
            # Apply correction with this alpha
            sig_val = signal[val_idx]
            y_corrected = y_pred_val + alpha * sig_val
            fold_errors.append(mean_squared_error(y_val, y_corrected))
        
        avg_rmse = np.sqrt(np.mean(fold_errors))
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_alpha = alpha
    
    print(f"  Optimal alpha: {best_alpha:.2f} (CV RMSE: {best_rmse:.4f})")
    return best_alpha

def apply_momentum_correction(y_pred, df_slice, alpha=0.3):
    """
    Apply momentum bias correction to reduce prediction delay.
    alpha controls how much correction to apply (0 = none, 1 = full momentum added).
    """
    signal = _build_correction_signal(df_slice)
    if signal is None:
        return y_pred
    
    correction = alpha * signal
    y_corrected = y_pred + correction
    
    print(f"  Momentum correction applied (alpha={alpha:.2f})")
    print(f"  Avg correction: {np.mean(correction):.4f}, Max: {np.max(np.abs(correction)):.4f}")
    
    return y_corrected

def main():
    input_path = 'data/processed/copper_features.csv'
    target_col = 'Target_Copper_Return_3M'
    price_target_col = 'Target_Copper_3M'
    
    print("Loading Data...")
    df = load_processed_data(input_path)
    
    # Split data
    data_known, data_future = prepare_data(df, target_col)
    drop_cols = [target_col, price_target_col]
    
    # ============================================================
    # KEY FIX: Remove raw price LEVELS from features
    # Raw levels (Copper_Price, Oil_Price, etc.) cause the model to
    # anchor on "current price" and just copy it forward = DELAY.
    # We only keep % changes, ratios, and technical indicators.
    # ============================================================
    raw_level_cols = [
        'Copper_Price', 'Gold_Price', 'Oil_Price', 'Aluminum_Price',
        'SP500', 'NVDA', 'DXY', 'AUD_USD', 'US_10yr_Yield',
        'China_Manuf_Prod', 'US_Auto_Sales', 'PPI_All_Commodities'
    ]
    # Keep these for price reconstruction, but don't feed them as features
    exclude_cols = drop_cols + [c for c in raw_level_cols if c in data_known.columns]
    
    X = data_known.drop(columns=exclude_cols)
    y = data_known[target_col]
    
    print(f"\nFeatures used ({len(X.columns)} total, raw levels excluded):")
    print(X.columns.tolist())
    
    test_size = 0.15
    split_idx = int(len(X) * (1 - test_size)) # Time series split
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Keep prices for reconstruction
    prices_test = data_known[price_target_col].iloc[split_idx:]
    current_prices_test = data_known['Copper_Price'].iloc[split_idx:]
    
    # Keep momentum columns for bias correction (from feature data, not from X)
    momentum_cols = [c for c in ['Copper_Accel_1M', 'Copper_Accel_3M', 
                                  'Momentum_Divergence_1v3', 'Momentum_Divergence_3v6',
                                  'Copper_Return_1M'] if c in data_known.columns]
    momentum_test = data_known[momentum_cols].iloc[split_idx:]
    momentum_train = data_known[momentum_cols].iloc[:split_idx:]
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # ============================================================
    # STEP 1: Train XGBoost (on ALL features)
    # ============================================================
    print("\n" + "="*60)
    print("STEP 1: Train XGBoost on ALL features")
    print("="*60)
    xgb_model = train_xgboost_model(X_train, y_train, tune=True)
    
    print("\nOptimizing momentum alpha for XGBoost...")
    xgb_alpha = optimize_alpha(xgb_model, X_train, y_train, momentum_train)
    xgb_prices = evaluate_and_plot(xgb_model, X_train, y_train, X_test, y_test, prices_test, current_prices_test, "XGBoost", momentum_correction=momentum_test, alpha_override=xgb_alpha)
    plot_feature_importance(xgb_model, X_train.columns, title="XGBoost Feature Importance")

    # Feature selection for downstream models
    print("\n" + "="*60)
    print("Feature Selection (Top 15 from XGBoost)")
    print("="*60)
    top_features = select_top_features(xgb_model, X_train.columns, top_n=15)
    X_train_selected = X_train[top_features]
    X_test_selected = X_test[top_features]

    # ============================================================
    # STEP 2: Train LightGBM (on selected features)
    # ============================================================
    print("\n" + "="*60)
    print("STEP 2: Train LightGBM on SELECTED features")
    print("="*60)
    lgbm_model = train_lightgbm_model(X_train_selected, y_train, tune=True)
    
    print("\nOptimizing momentum alpha for LightGBM...")
    lgbm_alpha = optimize_alpha(lgbm_model, X_train_selected, y_train, momentum_train)
    lgbm_prices = evaluate_and_plot(lgbm_model, X_train_selected, y_train, X_test_selected, y_test, prices_test, current_prices_test, "LightGBM", momentum_correction=momentum_test, alpha_override=lgbm_alpha)
    plot_feature_importance(lgbm_model, X_train_selected.columns, title="LightGBM Feature Importance")

    # ============================================================
    # STEP 3: Train ElasticNet (on selected features)
    # ============================================================
    print("\n" + "="*60)
    print("STEP 3: Train ElasticNet on SELECTED features")
    print("="*60)
    enet_model = train_elasticnet_model(X_train_selected, y_train)
    
    print("\nOptimizing momentum alpha for ElasticNet...")
    enet_alpha = optimize_alpha(enet_model, X_train_selected, y_train, momentum_train)
    enet_prices = evaluate_and_plot(enet_model, X_train_selected, y_train, X_test_selected, y_test, prices_test, current_prices_test, "ElasticNet", momentum_correction=momentum_test, alpha_override=enet_alpha)

    # ============================================================
    # STEP 4: Stack XGBoost + LightGBM + ElasticNet
    # ============================================================
    print("\n" + "="*60)
    print("STEP 4: Stacking Ensemble (XGB + LightGBM + ElasticNet)")
    print("="*60)
    stack = build_stacking_ensemble(xgb_model, lgbm_model, enet_model, X_train_selected, y_train)
    
    print("\nOptimizing momentum alpha for Stacked Ensemble...")
    stack_alpha = optimize_alpha(None, X_train_selected, y_train, momentum_train,
                                  predict_fn=lambda X: predict_stacked(stack, X))
    
    # Evaluate the stacking ensemble
    print(f"\n--- Evaluation: Stacked Ensemble ---")
    y_pred_stack = predict_stacked(stack, X_test_selected)
    y_pred_stack = apply_momentum_correction(y_pred_stack, momentum_test, alpha=stack_alpha)
    stack_prices = current_prices_test * (1 + y_pred_stack)
    
    rmse_ret = np.sqrt(mean_squared_error(y_test, y_pred_stack))
    mae_ret = mean_absolute_error(y_test, y_pred_stack)
    r2_ret = r2_score(y_test, y_pred_stack)
    rmse_price = np.sqrt(mean_squared_error(prices_test, stack_prices))
    mae_price = mean_absolute_error(prices_test, stack_prices)
    print(f"Metrics (Returns):\nRMSE: {rmse_ret:.4f}, MAE: {mae_ret:.4f}, R2: {r2_ret:.4f}")
    print(f"Metrics (Prices):\nRMSE: ${rmse_price:.2f}, MAE: ${mae_price:.2f}")

    # ============================================================
    # Comparison Plot (All Models)
    # ============================================================
    plt.figure(figsize=(14, 7))
    plt.plot(prices_test.index, prices_test, label='Actual Price', color='black', linewidth=2.5)
    plt.plot(prices_test.index, xgb_prices, label='XGBoost', linestyle='-.', color='blue', alpha=0.7)
    plt.plot(prices_test.index, lgbm_prices, label='LightGBM', linestyle='--', color='green', alpha=0.7)
    plt.plot(prices_test.index, enet_prices, label='ElasticNet', linestyle=':', color='orange', alpha=0.7)
    plt.plot(prices_test.index, stack_prices, label='Stacked Ensemble', linestyle='-', color='red', linewidth=2)
        
    plt.title('Copper Price Forecast Comparison (3-Month Horizon)')
    plt.ylabel('Price ($/lb)')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    output_fig = 'outputs/figures/model_comparison_all.png'
    os.makedirs(os.path.dirname(output_fig), exist_ok=True)
    plt.savefig(output_fig, dpi=150, bbox_inches='tight')
    print(f"\nValidation plot saved to {output_fig}")
    
    # ============================================================
    # Future Forecast (All 3 Models)
    # ============================================================
    if not data_future.empty:
        print("\nRetraining models on full dataset for Future Forecast...")
        
        X_future = data_future.drop(columns=exclude_cols)
        current_prices_future = data_future['Copper_Price']
        momentum_future = data_future[[c for c in momentum_cols if c in data_future.columns]]
        
        # Full data selected features
        X_selected = X[top_features]
        X_future_selected = X_future[top_features]
        
        # 1. Refit XGBoost on full data (all features)
        xgb_full = train_xgboost_model(X, y)
        future_ret_xgb = xgb_full.predict(X_future)
        future_ret_xgb = apply_momentum_correction(future_ret_xgb, momentum_future, alpha=xgb_alpha)
        future_price_xgb = current_prices_future * (1 + future_ret_xgb)
        
        # 2. Refit LightGBM on full data (selected features)
        lgbm_full = train_lightgbm_model(X_selected, y)
        future_ret_lgbm = lgbm_full.predict(X_future_selected)
        future_ret_lgbm = apply_momentum_correction(future_ret_lgbm, momentum_future, alpha=lgbm_alpha)
        future_price_lgbm = current_prices_future * (1 + future_ret_lgbm)
        
        # 3. Refit ElasticNet on full data (selected features)
        enet_full = train_elasticnet_model(X_selected, y)
        future_ret_enet = enet_full.predict(X_future_selected)
        future_ret_enet = apply_momentum_correction(future_ret_enet, momentum_future, alpha=enet_alpha)
        future_price_enet = current_prices_future * (1 + future_ret_enet)
        
        # 4. Refit Stacking on full data (selected features)
        stack_full = build_stacking_ensemble(xgb_model, lgbm_model, enet_model, X_selected, y)
        future_ret_stack = predict_stacked(stack_full, X_future_selected)
        future_ret_stack = apply_momentum_correction(future_ret_stack, momentum_future, alpha=stack_alpha)
        future_price_stack = current_prices_future * (1 + future_ret_stack)
        
        # Construct result DataFrame
        forecast = pd.DataFrame({
            'Current_Price': current_prices_future.values,
            'XGB_Price': future_price_xgb.values if hasattr(future_price_xgb, 'values') else future_price_xgb,
            'LightGBM_Price': future_price_lgbm.values if hasattr(future_price_lgbm, 'values') else future_price_lgbm,
            'ElasticNet_Price': future_price_enet.values if hasattr(future_price_enet, 'values') else future_price_enet,
            'Stacked_Price': future_price_stack.values if hasattr(future_price_stack, 'values') else future_price_stack
        }, index=X_future.index)
        
        print("\n--- Future Forecast (Next 3 Months) ---")
        print(forecast)
        
        output_csv = 'outputs/reports/forecast_3m_all_models.csv'
        forecast.to_csv(output_csv)
        print(f"Forecast saved to {output_csv}")
    else:
        print("No future data available.")

if __name__ == "__main__":
    main()
