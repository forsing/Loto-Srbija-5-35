from matplotlib.pylab import norm
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from qiskit_aer import AerSimulator
from qiskit_machine_learning.algorithms import NeuralNetworkRegressor
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.optimizers import COBYLA
from tqdm import tqdm
import random

from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import GradientDescent

from qiskit import QuantumCircuit, transpile

from qiskit.circuit import ParameterVector

from qiskit_machine_learning.kernels import FidelityQuantumKernel, TrainableFidelityQuantumKernel 
# QuantumKernelTrainer


import matplotlib.pyplot as plt

from sklearn import model_selection



import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



import qiskit
import qiskit_machine_learning
print()
print("Qiskit:", qiskit.__version__)
print("Qiskit Machine Learning:", qiskit_machine_learning.__version__)
print()
"""

"""

# pip list | grep qiskit
"""

"""


# =========================
# Seed za reproduktivnost
# =========================
SEED = 35
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED




# =========================
# Učitavanje CSV
# =========================
csv_path = '/data/loto5_153_k40.csv'
df = pd.read_csv(csv_path, header=None)


print()
print("Prvih 5 redova:")
print(df.head())
print()
"""
Prvih 5 redova:
   0   1   2   3   4  5
0  8  10  15  21  31  6
1  7  16  19  25  31  2
2  1   7  18  25  28  7
3  6   7  12  19  22  2
4  8  25  29  33  34  3
"""


print("Poslednjih 5 redova:")
print(df.tail())
print()
"""
Poslednjih 5 redova:

"""


data = df.values


# X - sve osim poslednje
X = data[:-1]

# y - sledeće pomereno za 1
y = data[1:]

# Normalizacija X i Y u [0, pi]
X = X / 35 * np.pi
y = y / 35 * np.pi

# Podela na trening i test skup
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.33, random_state=SEED
)



print()
print("len(X)")
print(len(X))
print()
"""

"""

print()
print("len(y)")
print(len(y))
print()
"""

"""



print()
print("len(X_train)")
print(len(X_train))
print()
"""

"""

print()
print("len(y_train)")
print(len(y_train))
print()
"""

"""



print()
print("len(X_test)")
print(len(X_test))
print()
"""

"""

print()
print("len(y_test)")
print(len(y_test))
print()
"""

"""




X2 = df.iloc[:, :-1].values  # prvih 5 brojeva
y_full = df.values          # svi 6 brojeva (5+1)

# Skaliranje
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X2).astype(np.float64)


num_qubits = X_scaled.shape[1]
print()
print("\nnum_qubits")
print(num_qubits, "\n")
print()
"""
num_qubits
5
"""



for i in range(6):  # 5 brojeva + dodatni broj

    y2 = y_full[:, i].astype(np.float64)
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y2.reshape(-1,1)).ravel()

    # plot data
    plt.plot(X_scaled, y_scaled, "bo")
    plt.title(f"Broj {i+1} - Podaci")
    plt.xlabel("Ulazni podaci (prvih 5 brojeva)")
    plt.ylabel(f"Izlazni podaci (broj {i+1})")
    plt.grid()
    # plt.show()



print()
print("len(X_scaled)")
print(len(X_scaled))
print()
"""

"""

print()
print("len(y_scaled)")
print(len(y_scaled))
print()
"""

"""




print()
print("len(X2)")
print(len(X2))
print()
"""

"""

print()
print("len(y2)")
print(len(y2))
print()
"""

"""




# Minimalni i maksimalni dozvoljeni brojevi po poziciji
min_val = [1, 2, 3, 4, 5, 1]
max_val = [31, 32, 33, 34, 35, 10]




from qiskit_machine_learning.circuit.library import qnn_circuit  # u 0.8.3 nema klase QNNCircuit, postoji funkcija qnn_circuit
qnn_qc, fm_params, anz_params = qnn_circuit(2)
print()
print("qnn_qc")
print(qnn_qc)
print()
"""

"""


qnn_qc.draw("mpl", style="clifford", fold=20)
plt.show()
"""

"""


print()
print("qnn_qc.num_qubits")
print(qnn_qc.num_qubits)
print()
"""
qnn_qc.num_qubits
2
"""



print()
print("qnn_qc.input_parameters")
print(fm_params)   # u 0.8.3 qnn_circuit vraća tuple (qc, fm_params, anz_params); QuantumCircuit nema .input_parameters
print()
"""
qnn_qc.input_parameters
ParameterView([ParameterVectorElement(x[0]), ParameterVectorElement(x[1])])
"""



print()
print("qnn_qc.weight_parameters")
print(anz_params)  # isto: QuantumCircuit nema .weight_parameters
print()
"""
qnn_qc.weight_parameters
ParameterView([ParameterVectorElement(θ[0]), ParameterVectorElement(θ[1]), ParameterVectorElement(θ[2]), ParameterVectorElement(θ[3]), ParameterVectorElement(θ[4]), ParameterVectorElement(θ[5]), ParameterVectorElement(θ[6]), ParameterVectorElement(θ[7])])
"""







# Create 2-qubit feature map
qc = QuantumCircuit(2)


print()
print("qc.num_qubits")
print(qc.num_qubits)
print()
"""
qc.num_qubits
2
"""


qc.draw("mpl", style="clifford", fold=20)
plt.show()
"""

"""



# Vectors of input and trainable user parameters
input_params = ParameterVector("x_par", 2)
training_params = ParameterVector("θ_par", 2)

# Create an initial rotation layer of trainable parameters
for i, param in enumerate(training_params):
    qc.ry(param, qc.qubits[i])

# Create a rotation layer of input parameters
for i, param in enumerate(input_params):
    qc.rz(param, qc.qubits[i])




class QuantumKernelTrainer:
    def __init__(self, quantum_kernel, optimizer):
        self.quantum_kernel = quantum_kernel
        self.optimizer = optimizer

    def fit(self, X, y):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError

    def get_params(self, deep=True):
        raise NotImplementedError

    def set_params(self, **params):
        raise NotImplementedError
    



quant_kernel = TrainableFidelityQuantumKernel(
    feature_map=qc,
    training_parameters=training_params,
)

"""

loss_func = ...

initial_point = ...
"""



optimizer = COBYLA(maxiter=1000)



qk_trainer = QuantumKernelTrainer(
                                quantum_kernel=quant_kernel,
                                #loss=loss_func,
                                optimizer=optimizer,
                                #initial_point=initial_point,
                                )


# qkt_results = qk_trainer.fit(X_train, y_train)

# optimized_kernel = qkt_results.quantum_kernel




################
# Estimator QNN

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.circuit.library import qnn_circuit  # u 0.8.3 nema klase QNNCircuit, postoji funkcija qnn_circuit

from qiskit_machine_learning.neural_networks import EstimatorQNN

num_qubits = 2

# Using the QNNCircuit:
# Create a parametrized 2 qubit circuit 
# composed of the default ZZFeatureMap feature map
# and RealAmplitudes ansatz.
qnn_qc, fm_params, anz_params = qnn_circuit(num_qubits)

qnn = EstimatorQNN(
    circuit=qnn_qc,
    input_params=fm_params,    # u 0.8.3 mora eksplicitno (nema više QNNCircuit instance)
    weight_params=anz_params,
)

qnn.forward(input_data=[1, 2], weights=[1, 2, 3, 4, 5, 6, 7, 8])

# Explicitly specifying the ansatz and feature map:
feature_map = ZZFeatureMap(feature_dimension=num_qubits)
ansatz = RealAmplitudes(num_qubits=num_qubits)

qc = QuantumCircuit(num_qubits)
qc.compose(feature_map, inplace=True)
qc.compose(ansatz, inplace=True)



qc.draw("mpl", style="clifford", fold=20)
plt.show()
"""

"""


qnn = EstimatorQNN(
    circuit=qc,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters
)

qnn.forward(input_data=[1, 2], weights=[1, 2, 3, 4, 5, 6, 7, 8])




##################
# SamplerQNN

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.circuit.library import qnn_circuit  # u 0.8.3 nema klase QNNCircuit, postoji funkcija qnn_circuit
from qiskit_machine_learning.neural_networks import SamplerQNN

num_qubits = 2

# Define a custom interpret function that calculates 
# the parity of the bitstring
def parity(x):
    return f"{bin(x)}".count("1") % 2

# Example 1: Using the QNNCircuit class
# QNNCircuit automatically combines a feature map 
# and an ansatz into a single circuit
qnn_qc, fm_params, anz_params = qnn_circuit(num_qubits)

qnn = SamplerQNN(
    circuit=qnn_qc,  # Note that this is a QNNCircuit instance
    input_params=fm_params,    # u 0.8.3 mora eksplicitno (nema više QNNCircuit instance)
    weight_params=anz_params,
    interpret=parity,
    output_shape=2  # Reshape by the number of classical registers
)

# Do a forward pass with input data and custom weights
qnn.forward(input_data=[1, 2], weights=[1, 2, 3, 4, 5, 6, 7, 8])

# Example 2: Explicitly specifying the feature map and ansatz
# Create a feature map and an ansatz separately
feature_map = ZZFeatureMap(feature_dimension=num_qubits)
ansatz = RealAmplitudes(num_qubits=num_qubits)

# Compose the feature map and ansatz manually (otherwise done within QNNCircuit)
qc = QuantumCircuit(num_qubits)
qc.compose(feature_map, inplace=True)
qc.compose(ansatz, inplace=True)

qnn = SamplerQNN(
    circuit=qc,  # Note that this is a QuantumCircuit instance
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
    interpret=parity,
    output_shape=2  # Reshape by the number of classical registers
)

# Perform a forward pass with input data and weights
qnn.forward(input_data=[1, 2], weights=[1, 2, 3, 4, 5, 6, 7, 8])




#######################
# optimizer

import random
import numpy as np
from qiskit_machine_learning.optimizers import GradientDescent

def objective(x):
    if random.choice([True, False]):
        return None
    else:
        return (np.linalg.norm(x) - 1) ** 2

def grad(x):
    if random.choice([True, False]):
        return None
    else:
        # return 2 * (np.linalg.norm(x) - 1) * x/np.linalg.norm(x)
        return x

initial_point = np.random.normal(0, 1, size=(100,))

optimizer = GradientDescent(maxiter=20)
optimizer.start(x0=initial_point, fun=objective, jac=grad)

result = optimizer.create_result()




################



from qiskit_machine_learning.optimizers import GradientDescent

def learning_rate():
    power = 0.6
    constant_coeff = 0.1
    def power_law():
        n = 0
        while True:
            yield constant_coeff * (n ** power)
            n += 1

    return power_law()

def f(x):
    return (np.linalg.norm(x) - 1) ** 2

def grad_f(x):
    return 2 * (np.linalg.norm(x) - 1) * x / np.linalg.norm(x)

initial_point = np.array([1, 0.5, -0.2])

optimizer = GradientDescent(maxiter=100, learning_rate=learning_rate)
result = optimizer.minimize(fun=f, jac=grad_f, x0=initial_point)
print()
print(f"Found minimum {result.x} at a value")
print()
"""

"""



############################



import random
import numpy as np
from qiskit_machine_learning.optimizers import GradientDescent

def objective(x):
    if random.choice([True, False]):
        return None
    else:
        return (np.linalg.norm(x) - 1) ** 2

def grad(x):
    if random.choice([True, False]):
        return None
    else:
        return 2 * (np.linalg.norm(x) - 1) * x / np.linalg.norm(x)


initial_point = np.random.normal(0, 1, size=(100,))

optimizer = GradientDescent(maxiter=20)
optimizer.start(x0=initial_point, fun=objective, jac=grad)



result = optimizer.create_result()



###############





import numpy as np
from qiskit_machine_learning.optimizers import SPSA
from qiskit.circuit.library import PauliTwoDesign
from qiskit.primitives import StatevectorEstimator as Estimator   # u 2.x nema više V1 Estimator
from qiskit.quantum_info import SparsePauliOp

ansatz = PauliTwoDesign(2, reps=1, seed=2)
observable = SparsePauliOp("ZZ")
initial_point = np.random.random(ansatz.num_parameters)
estimator = Estimator()

def loss(x):
    # V2 PUB API: [(circuit, observables, parameter_values)]
    job = estimator.run([(ansatz, observable, [x])])
    return float(job.result()[0].data.evs[0])

spsa = SPSA(maxiter=300)
result = spsa.minimize(loss, x0=initial_point)
# To use the Hessian information, i.e. 2-SPSA, you can add second_order=True to the initializer of the SPSA class, the rest of the code remains the same.

two_spsa = SPSA(maxiter=300, second_order=True)
result = two_spsa.minimize(loss, x0=initial_point)
# The termination_checker can be used to implement a custom termination criterion.

import numpy as np
from qiskit_machine_learning.optimizers import SPSA

def objective(x):
    return np.linalg.norm(x) + .04*np.random.rand(1)

class TerminationChecker:

    def __init__(self, N : int):
        self.N = N
        self.values = []

    def __call__(self, nfev, parameters, value, stepsize, accepted) -> bool:
        self.values.append(value)

        if len(self.values) > self.N:
            last_values = self.values[-self.N:]
            pp = np.polyfit(range(self.N), last_values, 1)
            slope = pp[0] / self.N

            if slope > 0:
                return True
        return False

spsa = SPSA(maxiter=200, termination_checker=TerminationChecker(10))
result = spsa.minimize(objective, x0=[0.5, 0.5])
print()
print(f'SPSA completed after {result.nit} iterations')
print()
"""

"""




#####################



import numpy as np
from qiskit_machine_learning.optimizers import QNSPSA
from qiskit.circuit.library import PauliTwoDesign
from qiskit.primitives import StatevectorEstimator as Estimator, StatevectorSampler as Sampler   # u 2.x nema više V1 Estimator/Sampler
from qiskit.quantum_info import Pauli, SparsePauliOp

# problem setup
ansatz = PauliTwoDesign(2, reps=1, seed=2)
observable = SparsePauliOp(Pauli("ZZ"))   # V2 estimator radi sa SparsePauliOp
initial_point = np.random.random(ansatz.num_parameters)

# loss function
estimator = Estimator()

def loss(x):
    # V2 PUB API: [(circuit, observables, parameter_values)]
    result = estimator.run([(ansatz, observable, [x])]).result()
    return float(np.real(result[0].data.evs[0]))

# fidelity for estimation of the geometric tensor
sampler = Sampler()



##############


from qiskit_machine_learning.optimizers import UMDA
# from qiskit_machine_learning import QAOA
from qiskit.quantum_info import Pauli
from qiskit.primitives import StatevectorSampler as Sampler   # u 2.x nema više V1 Sampler

X = Pauli("X")
I = Pauli("I")
Z = Pauli("Z")

H2_op = SparsePauliOp.from_list([("II", -1.0)])  # Pauli ne podržava skalarno množenje; -1*I^I → SparsePauliOp

p = 2  # Toy example: 2 layers with 2 parameters in each layer: 4 variables

opt = UMDA(maxiter=100, size_gen=20)
# qaoa = QAOA(Sampler(), opt,reps=p)
# result = qaoa.compute_minimum_eigenvalue(operator=H2_op)
# If it is desired to modify the percentage of individuals considered to update the probabilistic model, then this code can be used. Here for example we set the 60% instead of the 50% predefined.

opt = UMDA(maxiter=100, size_gen=20, alpha = 0.6)
# qaoa = QAOA(Sampler(), opt,reps=p)
# result = qaoa.compute_minimum_eigenvalue(operator=H2_op)


####################
# Implements Pegasos Quantum Support Vector Classifier algorithm.


quantum_kernel = FidelityQuantumKernel(feature_map=ZZFeatureMap(feature_dimension=2))   # u 0.8.3 feature_map je obavezan

class PegasosQSVC:
    def __init__(self, quantum_kernel):
        self.quantum_kernel = quantum_kernel

    def fit(self, sample_train, label_train):
        raise NotImplementedError

    def predict(self, sample_test):
        raise NotImplementedError

pegasos_qsvc = PegasosQSVC(quantum_kernel=quantum_kernel)
# pegasos_qsvc.fit(sample_train, label_train) 
# pegasos_qsvc.predict(sample_test)



###############################
# Predikcija sledeće kombinacije (deterministička, frekvencijska)
# Loto: 5 glavnih iz 1..35 + 1 dopunski iz 1..10
###############################
from collections import Counter

print()
print("=== Predikcija sledeće kombinacije (Loto 5/35 + dopunski 1..10) ===")

# Frekvencije glavnih brojeva iz svih prvih 5 kolona
main_vals = df.iloc[:, :5].values.ravel().astype(int).tolist()
freq_main = Counter(main_vals)
predicted_main = sorted(
    [n for n, _ in freq_main.most_common() if 1 <= n <= 35][:5]
)

# Frekvencija dopunskog broja (6. kolona) — opseg 1..10
bonus_vals = df.iloc[:, 5].values.astype(int).tolist()
freq_bonus = Counter(bonus_vals)
predicted_bonus = next(n for n, _ in freq_bonus.most_common() if 1 <= n <= 10)

prediction = predicted_main + [predicted_bonus]
print("Glavnih 5:", predicted_main)
print("Dopunski :", predicted_bonus)
print("Sve     :", prediction)
print()

"""
Sve     : 2, x, y, z, 28, 6
"""




"""
Vise je Qiskit / Qiskit Machine Learning 
demo-skripta za loto podatke, 
nego jedan čist završni prediktor. 
Radi na Srbija Loto 5 od 35 + dopunski 1 do 10.


Prvo učita CSV bez header-a, prikaže prvih i poslednjih 5 redova, 
pa priprema podatke:

X = data[:-1]
y = data[1:]
Znači koristi princip vremenske serije: 
jedan red kao ulaz, sledeći red kao cilj.

Zatim radi normalizaciju u opseg [0, pi], 
što je tipično za kvantna kola 
jer se brojevi koriste kao uglovi rotacija.

Kvantne metode u fajlu
U fajlu su demonstrirane više Qiskit ML tehnike:

qnn_circuit() 
EstimatorQNN primer: kvantna neuronska mreža koja vraća očekivanu vrednost observable-a.
SamplerQNN primer: kvantna mreža koja radi preko uzorkovanja bitstringova.
ZZFeatureMap: mapira klasične ulazne podatke u kvantno stanje.
RealAmplitudes: varijacioni ansatz, trenabilni deo kola.
TwoLocal / ručno komponovana kola: koristi se za pravljenje parametarskog kvantnog kola.
FidelityQuantumKernel i TrainableFidelityQuantumKernel: kvantni kernel pristup, više za klasifikaciju/SVM stil nego direktnu predikciju.
PegasosQSVC demo-klasa: kostur za kvantni SVM, nije stvarno treniran.
optimizatori: COBYLA, GradientDescent, SPSA, QNSPSA, UMDA.


Na kraju je dodata jednostavna deterministička predikcija:

računa frekvencije svih glavnih brojeva iz prvih 5 kolona,
bira 5 najčešćih brojeva u opsegu 1..35,
računa najčešći dopunski broj iz 6. kolone u opsegu 1..10,
ispisuje kombinaciju.

To nije kvantna predikcija, 
nego brz deterministički frekvencijski baseline. 
Kvantni delovi u fajlu su više API/demo blokovi za proveru metoda.
"""
