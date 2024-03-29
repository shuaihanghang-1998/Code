package com.cmj.oa.dao;

import com.cmj.oa.entity.Employee;
import org.apache.ibatis.annotations.Param;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository("employeeDao")
public interface EmployeeDao {

    void insert(Employee employee);

    void update(Employee employee);

    void delete(String sn);

    Employee select(String sn);

    String selectNameBySn(String sn);

    List<Employee> selectAll();

    List<Employee> selectByDepartmentAndPost(@Param("dsn") String dsn,@Param("post") String post);
}
