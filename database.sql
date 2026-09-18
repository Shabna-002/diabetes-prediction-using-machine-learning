CREATE DATABASE IF NOT EXISTS diabetes_prediction_db;
USE diabetes_prediction_db;

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100) NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_name VARCHAR(100) NULL,
    model_used VARCHAR(50) NULL,
    pregnancies INT NOT NULL,
    glucose FLOAT NOT NULL,
    blood_pressure FLOAT NOT NULL,
    skin_thickness FLOAT NOT NULL,
    insulin FLOAT NOT NULL,
    bmi FLOAT NOT NULL,
    diabetes_pedigree FLOAT NOT NULL,
    age INT NOT NULL,
    prediction TINYINT NOT NULL,
    probability FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_prediction_created_at ON predictions(created_at);
CREATE INDEX idx_prediction_result ON predictions(prediction);
CREATE INDEX idx_users_username ON users(username);
