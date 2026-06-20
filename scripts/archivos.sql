CREATE TABLE archivos_consumo (
    id_archivo INT AUTO_INCREMENT PRIMARY KEY,
    id_usuario INT NOT NULL,
    nombre_archivo VARCHAR(255) NOT NULL,
    fecha_importacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_curvas INT NOT NULL,
    estadisticas_json LONGTEXT NULL,
    FOREIGN KEY (id_usuario)
        REFERENCES usuarios(id_usuario)
);