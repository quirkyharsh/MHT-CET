import os
import sys

from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score

from xgboost import XGBRegressor

from catboost import CatBoostRegressor

from src.exception import CustomException

from src.logger import logging

from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:

    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):

        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(
        self,
        train_array,
        test_array
    ):

        try:

            logging.info(
                "Splitting training and testing input data"
            )

            # ---------------- TRAIN TEST SPLIT ----------------

            X_train = train_array[:, :-1]

            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]

            y_test = test_array[:, -1]

            logging.info("Data split completed")


            # ---------------- MODELS ----------------

            models = {

                "Linear Regression": LinearRegression(),

                "Ridge": Ridge(),

                "Lasso": Lasso(),

                "Random Forest": RandomForestRegressor(
                    random_state=42,
                    n_jobs=-1
                ),

                "XGBoost Regressor": XGBRegressor(
                    random_state=42,
                    n_jobs=-1
                ),

                "CatBoost Regressor": CatBoostRegressor(
                    verbose=False,
                    random_state=42
                )
            }


            # ---------------- HYPERPARAMETERS ----------------

            params = {

                "Linear Regression": {},

                "Ridge": {

                    'alpha': [0.1, 1.0, 10.0]
                },

                "Lasso": {

                    'alpha': [0.001, 0.01, 0.1]
                },

                "Random Forest": {

                    'n_estimators': [200, 400],

                    'max_depth': [10, 20, None],

                    'min_samples_leaf': [1, 2]
                },

                "XGBoost Regressor": {

                    'n_estimators': [200, 400],

                    'max_depth': [6, 10],

                    'learning_rate': [0.05, 0.1]
                },

                "CatBoost Regressor": {

                    'iterations': [300, 600],

                    'depth': [6, 10],

                    'learning_rate': [0.03, 0.1]
                }
            }


            # ---------------- MODEL EVALUATION ----------------

            model_report = evaluate_models(

                X_train=X_train,

                y_train=y_train,

                X_test=X_test,

                y_test=y_test,

                models=models,

                param=params
            )

            logging.info(
                f"Model Report : {model_report}"
            )


            # ---------------- BEST MODEL ----------------

            best_model_score = max(
                sorted(model_report.values())
            )

            best_model_name = list(
                model_report.keys()
            )[
                list(model_report.values()).index(
                    best_model_score
                )
            ]

            best_model = models[best_model_name]

            logging.info(
                f"Best Model Found : {best_model_name}"
            )

            logging.info(
                f"Best Model R2 Score : {best_model_score}"
            )


            # ---------------- VALIDATION ----------------

            if best_model_score < 0.6:

                raise CustomException(
                    "No best model found"
                )


            # ---------------- TRAIN BEST MODEL ----------------

            best_model.fit(
                X_train,
                y_train
            )


            # ---------------- SAVE MODEL ----------------

            save_object(

                file_path=self.model_trainer_config.trained_model_file_path,

                obj=best_model
            )

            logging.info(
                "Best model saved successfully"
            )


            # ---------------- PREDICTION ----------------

            predicted = best_model.predict(
                X_test
            )

            r2 = r2_score(
                y_test,
                predicted
            )

            logging.info(
                f"Final R2 Score : {r2}"
            )

            return r2


        except Exception as e:

            raise CustomException(e, sys)