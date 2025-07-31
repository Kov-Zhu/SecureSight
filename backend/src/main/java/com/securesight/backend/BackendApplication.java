package com.securesight.backend;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.data.mongo.MongoDataAutoConfiguration;
import org.springframework.boot.autoconfigure.mongo.MongoAutoConfiguration;

import io.github.cdimascio.dotenv.Dotenv;

@SpringBootApplication(exclude = {
        MongoAutoConfiguration.class,
        MongoDataAutoConfiguration.class
})
public class BackendApplication {

    @Value("${spring.data.mongodb.uri}")
    private String mongoUri;

    public static void main(String[] args) {
        Dotenv dotenv = Dotenv.configure().load();
        System.setProperty("MONGODB_PASSWORD", dotenv.get("MONGODB_PASSWORD"));

        SpringApplication.run(BackendApplication.class, args);
    }

    // Add a method to log the mongoUri after the application context is initialized
    @org.springframework.context.event.EventListener(org.springframework.boot.context.event.ApplicationReadyEvent.class)
    public void logMongoUri() {
        System.out.println("LogMongoDB URI: " + mongoUri);
    }
}
