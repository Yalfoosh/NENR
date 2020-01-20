from enum import Enum, auto
import re

loss_pattern = r"(l(oss)?)"
accuracy_pattern = r"(acc(u(r(acy)?)?)?)"

training_pattern = r"(t(r(ain)?)?)"
validation_pattern = r"(val(i(d(at(ion)?)?)?)?)"


metrics_regexi =\
    {
        "loss": re.compile(r"(?i)((\s*{}\s*_)?\s*{}\s*)".format(training_pattern, loss_pattern)),
        "val_loss": re.compile(r"(?i)((\s*{}\s*_)?\s*{}\s*)".format(validation_pattern, loss_pattern)),
        "acc": re.compile(r"(?i)((\s*{}\s*_)?\s*{}\s*)".format(training_pattern, accuracy_pattern)),
        "val_acc": re.compile(r"(?i)((\s*{}\s*_)?\s*{}\s*)".format(validation_pattern, accuracy_pattern))
    }


class Metric(Enum):
    Loss = auto()
    Accuracy = auto()
    ValidationLoss = auto()
    ValidationAccuracy = auto()

    metric_to_name =\
        {
            Loss: "loss",
            Accuracy: "acc",
            ValidationLoss: "val_loss",
            ValidationAccuracy: "val_acc"
        }

    @staticmethod
    def enum_to_name(metric_name: str):
        for name in metrics_regexi:
            if metrics_regexi[name].match(metric_name):
                return name

    @staticmethod
    def get_key(metric):
        return Metric.metric_to_name[metric]
