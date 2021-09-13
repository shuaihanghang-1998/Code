package com.cmj.oa.biz;

import com.cmj.oa.entity.Employee;

public interface GlobalBiz {

    Employee login(String sn,String password);

    void changePassword(Employee employee);



}
