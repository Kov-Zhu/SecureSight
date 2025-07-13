package com.securesight.backend.controller;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.securesight.backend.model.Alarm;
import com.securesight.backend.repository.AlarmRepository;

@RestController
@RequestMapping("/api/alarms")
public class AlarmController {

    @Autowired
    private AlarmRepository alarmRepository;

    @GetMapping
    public List<Alarm> getAllAlarms() {
        return alarmRepository.findAll();
    }

    @PostMapping
    public Alarm createAlarm(@RequestBody Alarm alarm) {
        return alarmRepository.save(alarm);
    }
    
}
