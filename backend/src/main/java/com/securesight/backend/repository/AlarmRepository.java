package com.securesight.backend.repository;

import org.springframework.data.jpa.repository.JpaRepository;

import com.securesight.backend.model.Alarm;

public interface AlarmRepository extends JpaRepository<Alarm, Long> {
}
