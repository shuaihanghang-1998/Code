package com.cmj.oa.biz;

import com.cmj.oa.entity.Department;

import java.util.List;

public interface DepartmentBiz {

    void add(Department department);
    void edit(Department department);
    void remove(String sn);
    Department get(String sn);
    List<Department> getAll();
    List<Department> getdlist();
}
