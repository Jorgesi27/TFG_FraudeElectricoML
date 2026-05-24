-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Servidor: 127.0.0.1
-- Tiempo de generación: 24-05-2026 a las 15:42:25
-- Versión del servidor: 10.4.32-MariaDB
-- Versión de PHP: 8.2.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Base de datos: `tfg_fraudeelectrico`
--

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `archivos_consumo`
--

CREATE TABLE `archivos_consumo` (
  `id_archivo` int(11) NOT NULL,
  `id_usuario` int(11) NOT NULL,
  `nombre_archivo` varchar(255) NOT NULL,
  `fecha_importacion` timestamp NOT NULL DEFAULT current_timestamp(),
  `total_curvas` int(11) NOT NULL,
  `estadisticas_json` longtext DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_general_ci;

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `curvas_consumo`
--

CREATE TABLE `curvas_consumo` (
  `id_curva` int(11) NOT NULL,
  `id_archivo` int(11) NOT NULL,
  `identificador_curva` varchar(100) NOT NULL,
  `datos_consumo` longtext NOT NULL,
  `fecha_registro` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_general_ci;

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `predicciones`
--

CREATE TABLE `predicciones` (
  `id_prediccion` int(11) NOT NULL,
  `id_curva` int(11) NOT NULL,
  `tipo_modelo` enum('xgboost','adaptive_random_forest') NOT NULL,
  `tipo_prediccion` enum('historica','tiempo_real') NOT NULL,
  `resultado_prediccion` tinyint(1) NOT NULL,
  `probabilidad_fraude` float NOT NULL,
  `fecha_prediccion` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_general_ci;

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `usuarios`
--

CREATE TABLE `usuarios` (
  `id_usuario` int(11) NOT NULL,
  `nombre_usuario` varchar(50) NOT NULL,
  `password_hash` varchar(255) NOT NULL,
  `rol` enum('operador','administrador') NOT NULL,
  `activo` tinyint(1) NOT NULL DEFAULT 1,
  `fecha_creacion` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_general_ci;

--
-- Volcado de datos para la tabla `usuarios`
--

INSERT INTO `usuarios` (`id_usuario`, `nombre_usuario`, `password_hash`, `rol`, `activo`, `fecha_creacion`) VALUES
(1, 'operador', '$2b$12$pkOzyZlb/MVPrSRAV16SUumzFQQx/9tI28Chp4we21lLagfMr/Juq', 'operador', 1, '2026-05-02 21:19:53'),
(2, 'admin', '$2b$12$BvOuLyYlEJDQJXStv68Sp.NEPxZb2wMbPQ.Qkb83yPNCzQxxDi3nS', 'administrador', 1, '2026-05-02 21:19:53');

--
-- Índices para tablas volcadas
--

--
-- Indices de la tabla `archivos_consumo`
--
ALTER TABLE `archivos_consumo`
  ADD PRIMARY KEY (`id_archivo`),
  ADD KEY `id_usuario` (`id_usuario`);

--
-- Indices de la tabla `curvas_consumo`
--
ALTER TABLE `curvas_consumo`
  ADD PRIMARY KEY (`id_curva`),
  ADD KEY `id_archivo` (`id_archivo`);

--
-- Indices de la tabla `predicciones`
--
ALTER TABLE `predicciones`
  ADD PRIMARY KEY (`id_prediccion`),
  ADD KEY `fk_prediccion_curva` (`id_curva`);

--
-- Indices de la tabla `usuarios`
--
ALTER TABLE `usuarios`
  ADD PRIMARY KEY (`id_usuario`),
  ADD UNIQUE KEY `nombre_usuario` (`nombre_usuario`);

--
-- AUTO_INCREMENT de las tablas volcadas
--

--
-- AUTO_INCREMENT de la tabla `archivos_consumo`
--
ALTER TABLE `archivos_consumo`
  MODIFY `id_archivo` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT de la tabla `curvas_consumo`
--
ALTER TABLE `curvas_consumo`
  MODIFY `id_curva` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT de la tabla `predicciones`
--
ALTER TABLE `predicciones`
  MODIFY `id_prediccion` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT de la tabla `usuarios`
--
ALTER TABLE `usuarios`
  MODIFY `id_usuario` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- Restricciones para tablas volcadas
--

--
-- Filtros para la tabla `archivos_consumo`
--
ALTER TABLE `archivos_consumo`
  ADD CONSTRAINT `archivos_consumo_ibfk_1` FOREIGN KEY (`id_usuario`) REFERENCES `usuarios` (`id_usuario`);

--
-- Filtros para la tabla `curvas_consumo`
--
ALTER TABLE `curvas_consumo`
  ADD CONSTRAINT `curvas_consumo_ibfk_1` FOREIGN KEY (`id_archivo`) REFERENCES `archivos_consumo` (`id_archivo`);

--
-- Filtros para la tabla `predicciones`
--
ALTER TABLE `predicciones`
  ADD CONSTRAINT `fk_prediccion_curva` FOREIGN KEY (`id_curva`) REFERENCES `curvas_consumo` (`id_curva`) ON DELETE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
