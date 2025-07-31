// TODO The test has not been completed.

package com.securesight.backend;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.test.autoconfigure.data.mongo.DataMongoTest;
import org.springframework.data.mongodb.core.MongoTemplate;

@DataMongoTest()
public class MongoConnectionTest {

    @Value("${spring.data.mongodb.uri}")
    private String mongoUri;

    @Autowired
    private MongoTemplate mongoTemplate;

    @Test
    void testConnection() {
        System.out.println("LogMongoDB URI: " + mongoUri);
        Assertions.assertNotNull(mongoTemplate);
    }
}