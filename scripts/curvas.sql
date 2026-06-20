CREATE TABLE curvas_consumo (
    id_curva INT AUTO_INCREMENT PRIMARY KEY,
    id_archivo INT NOT NULL,
    identificador_curva VARCHAR(100) NOT NULL,
    datos_consumo LONGTEXT NOT NULL,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (id_archivo)
        REFERENCES archivos_consumo(id_archivo)
);