package com.securesight.backend.controller;

import com.securesight.backend.model.Alarm;
import com.securesight.backend.repository.AlarmRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

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
