import os
import sys

import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class BranchNotOfferedException(Exception):
    """Raised when the requested branch has no historical record at the requested college."""
    pass


class PredictPipeline:

    def __init__(self):

        pass


    def predict(self, features):

        try:

            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            data_path = os.path.join("artifacts", "data.csv")

            model = load_object(file_path=model_path)

            preprocessor = load_object(file_path=preprocessor_path)

            history = pd.read_csv(data_path)

            college_name = features["college_name"].iloc[0]
            branch = features["branch"].iloc[0]
            seat_type = features["seat_type"].iloc[0]
            score_type = features["score_type"].iloc[0]

            # ---------------- VALIDATE COLLEGE + BRANCH EXIST TOGETHER ----------------
            # If this branch was never recorded at this college for ANY
            # category/score_type, it almost certainly isn't offered there.
            # Predicting anyway would silently fabricate a cutoff for a
            # branch-college combination that doesn't exist, so we refuse
            # instead of falling back to a global average.

            college_branch_match = history[
                (history["college_name"] == college_name) &
                (history["branch"] == branch)
            ]

            if len(college_branch_match) == 0:
                raise BranchNotOfferedException(
                    f"'{branch}' does not appear to be offered at '{college_name}'. "
                    f"Please choose a branch that exists for this college."
                )

            # ---------------- AUTO-FILL 'count' ----------------
            # 'count' (candidates/seats recorded for this exact combination)
            # is a real feature the model was trained on, but a prospective
            # student has no way of knowing this value themselves. We look
            # up the historical average for the exact (college, branch,
            # seat_type, score_type) combination, and if that precise
            # combination wasn't recorded, fall back to the average across
            # just this college+branch (which we've already confirmed exists)
            # rather than the entire dataset — a much closer estimate.

            exact_match = college_branch_match[
                (college_branch_match["seat_type"] == seat_type) &
                (college_branch_match["score_type"] == score_type)
            ]

            if len(exact_match) > 0:
                avg_count = exact_match["count"].mean()
            else:
                avg_count = college_branch_match["count"].mean()

            features = features.copy()
            features["count"] = avg_count

            data_scaled = preprocessor.transform(features)

            preds = model.predict(data_scaled)

            return preds

        except BranchNotOfferedException:

            raise

        except Exception as e:

            raise CustomException(e, sys)


class CustomData:

    def __init__(
        self,
        college_name: str,
        score_type: str,
        seat_type: str,
        branch: str
    ):

        self.college_name = college_name

        self.score_type = score_type

        self.seat_type = seat_type

        self.branch = branch


    def get_data_as_data_frame(self):

        try:

            custom_data_input_dict = {

                "college_name": [self.college_name],

                "score_type": [self.score_type],

                "seat_type": [self.seat_type],

                "branch": [self.branch],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:

            raise CustomException(e, sys)
