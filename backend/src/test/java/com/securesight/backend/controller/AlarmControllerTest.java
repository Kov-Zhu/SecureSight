package com.securesight.backend.controller;

import java.util.Collections;

import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.securesight.backend.model.Alarm;
import com.securesight.backend.repository.AlarmRepository;

@WebMvcTest(AlarmController.class)
public class AlarmControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockitoBean
    private AlarmRepository alarmRepository;

    @Test
    void testGetAllAlarms() throws Exception {
        Mockito.when(alarmRepository.findAll()).thenReturn(Collections.emptyList());
        mockMvc.perform(get("/api/alarms"))
                .andExpect(status().isOk())
                .andExpect(content().json("[]"));
    }

    @Test
    void testCreateAlarm() throws Exception {
        Alarm alarm = new Alarm();
        alarm.setId(1L);
        alarm.setTimestamp(java.time.LocalDateTime.now());
        alarm.setImageUrl("http://example.com/image.jpg");
        alarm.setDescription("Test Description");
        alarm.setSeverity("HIGH");
        Mockito.when(alarmRepository.save(Mockito.any(Alarm.class))).thenReturn(alarm);

        String alarmJson = String.format("{\"timestamp\":\"%s\",\"imageUrl\":\"http://example.com/image.jpg\",\"description\":\"Test Description\",\"severity\":\"HIGH\"}", alarm.getTimestamp().toString());
        String expectedJson = String.format("{\"id\":1,\"timestamp\":\"%s\",\"imageUrl\":\"http://example.com/image.jpg\",\"description\":\"Test Description\",\"severity\":\"HIGH\"}", alarm.getTimestamp().toString());

        mockMvc.perform(post("/api/alarms")
                .contentType(MediaType.APPLICATION_JSON)
                .content(alarmJson))
                .andExpect(status().isOk())
                .andExpect(content().json(expectedJson));
    }
}
