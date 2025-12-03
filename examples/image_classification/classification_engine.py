import logging
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .cnn import Net
from .conversion import Converter
from .ic_training import DataManger, execute_ic_training

from fl_main.agent.client import Client


class TrainingMetaData:
    # Cantidad de datos de entrenamiento usados por ronda
    # Se usará para el promedio ponderado (FedAvg)
    # Debe ser un número natural > 0
    num_training_data = 8000

def init_models() -> Dict[str,np.array]:
    """
    Devuelve las plantillas de los modelos (en un dict) para indicar la estructura.
    Los modelos no necesitan estar entrenados.
    :return: Dict[str,np.array]
    """
    net = Net()
    return Converter.cvtr().convert_nn_to_dict_nparray(net)

def training(models: Dict[str,np.array], init_flag: bool = False) -> Dict[str,np.array]:
    """
    Función de marcador de posición para cada aplicación de ML.
    Devuelve los modelos entrenados.
    Nota: cada modelo debe descomponerse a arreglos de NumPy.
    La lógica debe ser de la forma: modelos -- entrenamiento --> nuevos modelos locales
    :param models: Dict[str,np.array]
    :param init_flag: bool - True si es el paso de inicialización.
    False si es un paso de entrenamiento real.
    :return: Dict[str,np.array] - modelos entrenados
    """
    # devolver plantillas de modelos para indicar la estructura
    # Este modelo no necesariamente está realmente entrenado
    if init_flag:
        # Preparar los datos de entrenamiento
        # num de muestras / 4 = umbral de entrenamiento debido al tamaño de batch
        DataManger.dm(int(TrainingMetaData.num_training_data / 4))
        return init_models()

    # Realizar el entrenamiento de ML
    logging.info(f'--- Training ---')

    # Crear una CNN a partir de los modelos globales (cluster)
    net = Converter.cvtr().convert_dict_nparray_to_nn(models)

    # Definir la función de pérdida y el optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # modelos -- entrenamiento --> nuevos modelos locales
    trained_net = execute_ic_training(DataManger.dm(), net, criterion, optimizer)
    models = Converter.cvtr().convert_nn_to_dict_nparray(trained_net)
    return models

def compute_performance(models: Dict[str,np.array], testdata, is_local: bool) -> float:
    """
    Dado un conjunto de modelos y un dataset de prueba, calcular el desempeño de los modelos.
    :param models:
    :param testdata:
    :return:
    """
    # Convertir arreglos de NumPy a una CNN
    net = Converter.cvtr().convert_dict_nparray_to_nn(models)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in DataManger.dm().testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = float(correct) / total

    mt = 'local'
    if not is_local:
        mt = 'Global'

    print(f'Accuracy of the {mt} model with the 10000 test images: {100 * acc} %%')

    return acc

def judge_termination(training_count: int = 0, gm_arrival_count: int = 0) -> bool:
    """
    Decidir si finaliza el proceso de entrenamiento y sale de la plataforma FL.
    :param training_count: int - número de entrenamientos realizados
    :param gm_arrival_count: int - número de veces que recibió modelos globales
    :return: bool - True si continúa el bucle de entrenamiento; False si se detiene
    """
    # Podría llamarse a un rastreador de desempeño para verificar si los modelos actuales cumplen la métrica requerida
    return True

def prep_test_data():
    # Preparar y devolver el dataset de prueba (placeholder)
    testdata = 0
    return testdata

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('--- This is a demo of Image Classification with Federated Learning ---')

    fl_client = Client()
    logging.info(f'--- Your IP is {fl_client.agent_ip} ---')

    # Crear un conjunto de modelos plantilla (para indicar las formas)
    initial_models = training(dict(), init_flag=True)

    # Enviar modelos iniciales
    fl_client.send_initial_model(initial_models)

    # Iniciar el cliente FL
    fl_client.start_fl_client()

    training_count = 0
    gm_arrival_count = 0
    while judge_termination(training_count, gm_arrival_count):

        # Esperar modelos globales (modelos base)
        global_models = fl_client.wait_for_global_model()
        gm_arrival_count += 1

        # Evaluación del modelo global (id, exactitud)
        global_model_performance_data = compute_performance(global_models, prep_test_data(), False)

        # Entrenamiento
        models = training(global_models)
        training_count += 1
        logging.info(f'--- Training Done ---')

        # Evaluación del modelo local (id, exactitud)
        accuracy = compute_performance(models, prep_test_data(), True)
        fl_client.send_trained_model(models, int(TrainingMetaData.num_training_data), accuracy)
