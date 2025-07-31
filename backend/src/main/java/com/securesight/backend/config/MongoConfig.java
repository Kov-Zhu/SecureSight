package com.securesight.backend.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.mongodb.core.MongoTemplate;

import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;

import io.github.cdimascio.dotenv.Dotenv;

@Configuration
public class MongoConfig {

    @Bean
    public MongoClient mongoClient() {
        Dotenv dotenv = Dotenv.configure().load();
        String password = dotenv.get("MONGODB_PASSWORD");
        String uri = "mongodb+srv://admin:" + password +
                "@cluster0.o7zag.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0";
        return MongoClients.create(uri);
    }

    @Bean
    public MongoTemplate mongoTemplate(MongoClient mongoClient) {
        return new MongoTemplate(mongoClient, "SecureSight");
    }
}