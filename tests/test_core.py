import numpy as np
from src.preprocessing import db_to_linear
from src.rvi_calculation import calculate_rvi
from src.anomaly_detection import classify_risk
def test_db(): assert np.isclose(db_to_linear(np.array([0.]))[0],1)
def test_rvi_linear(): assert np.isclose(calculate_rvi([1],[.25],units='linear')[0],.8)
def test_risk(): assert list(classify_risk(np.array([-45,-35,-25,-15,-5])))==['Critical','High Risk','Moderate Risk','Watch','Normal']
