package com.cmj.oa.biz;

import com.cmj.oa.entity.Employee;

import java.util.List;

public interface EmployeeBiz {

    void add(Employee employee);
    void edit(Employee employee);
    void remove(String sn);
    Employee get(String sn);
    String getEmployeeName(String sn);
    List<Employee> getAll();
}
