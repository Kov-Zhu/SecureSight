package com.securesight.backend;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import io.github.cdimascio.dotenv.Dotenv;

@SpringBootApplication
public class BackendApplication {

    public static void main(String[] args) {
        Dotenv dotenv = Dotenv.configure().load();
        System.setProperty("MONGODB_PASSWORD", dotenv.get("MONGODB_PASSWORD"));
        SpringApplication.run(BackendApplication.class, args);
    }
}
