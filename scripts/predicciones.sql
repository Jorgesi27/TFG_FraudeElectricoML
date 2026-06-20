CREATE TABLE predicciones (
    id_prediccion INT AUTO_INCREMENT PRIMARY KEY,
    id_curva INT NOT NULL,
    tipo_modelo ENUM(
        'xgboost',
        'adaptive_random_forest'
    ) NOT NULL,
    tipo_prediccion ENUM(
        'historica',
        'tiempo_real'
    ) NOT NULL,
    resultado_prediccion TINYINT(1) NOT NULL,
    probabilidad_fraude FLOAT NOT NULL,
    fecha_prediccion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (id_curva)
        REFERENCES curvas_consumo(id_curva)
);