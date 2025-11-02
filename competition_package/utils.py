import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from dataclasses import dataclass
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


@dataclass
class DataPoint:
    seq_ix: int
    step_in_seq: int
    need_prediction: bool
    #
    state: np.ndarray


class PredictionModel:
    def predict(self, data_point: DataPoint) -> np.ndarray:
        # return current state as dummy prediction
        print("HIHIOHOHOHOo")
        return data_point.state


class ScorerStepByStep:
    def __init__(self, dataset_path: str):
        self.dataset = pd.read_csv(dataset_path)

        # Calc feature dimension: first 3 columns are seq_ix, step_in_seq & need_prediction
        self.dim = self.dataset.shape[1] - 3
        self.features = self.dataset.columns[3:]

    def plot(self,Y_train, predicted):
        plt.plot(Y_train, label='Actual Close')
        plt.plot(predicted, label='Predicted Close')
        plt.savefig("plot/graph_img.png")
        plt.show()

    def score(self, model: PredictionModel) -> dict:
        predictions = []
        targets = []

        next_prediction = None

        for row in tqdm(self.dataset.values):
            seq_ix = row[0]
            step_in_seq = row[1]
            need_prediction = row[2]
            new_state = row[3:]  # the rest is state vector
            #
            if next_prediction is not None:
                predictions.append(next_prediction)
                targets.append(new_state)
            #
            data_point = DataPoint(seq_ix, step_in_seq, need_prediction, new_state)
            next_prediction = model.predict(data_point)
            #print(row,next_prediction)

            self.check_prediction(data_point, next_prediction)
        
        pred_arr = np.array(predictions)
        targ_arr = np.array(targets)
        self.plot(targ_arr[0],pred_arr[0])
        # report metrics
        return self.calc_metrics(pred_arr,targ_arr)

    def check_prediction(self, data_point: DataPoint, prediction: np.ndarray):
        if not data_point.need_prediction:
            if prediction is not None:
                raise ValueError(f"Prediction is not needed for {data_point}")
            return

        if prediction is None:
            raise ValueError(f"Prediction is required for {data_point}")

        if prediction.shape[0] != self.dim:
            raise ValueError(
                f"Prediction has wrong shape: {prediction.shape[0]} != {self.dim}"
            )

    def calc_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> dict:
        scores = {}
        for ix_feature, feature in enumerate(self.features):
            scores[feature] = r2_score(
                targets[:, ix_feature], predictions[:, ix_feature]
            )
        scores["mean_r2"] = np.mean(list(scores.values()))
        return scores
