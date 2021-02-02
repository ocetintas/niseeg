import innvestigate
import innvestigate.utils as iutils
from keras.models import load_model
from dataset.DEAP_keras import DEAP


checkpoint = ""
subject = 10
datset = DEAP(subject)
sample_idx = 0

eeg = dataset[idx]["eeg"]
fl = dataset[idx]["face"]
concat = np.concatenate((eeg, fl), axis=-1)

model_without_softmax = load_model(checkpoint)

# Creating an analyzer
gradient_analyzer = innvestigate.create_analyzer("lrp.z", model_without_softmax)

# Applying the analyzer
analysis = gradient_analyzer.analyze(eeg)