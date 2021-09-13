package com.cmj.oa.controller;

import com.cmj.oa.biz.ClaimVoucherBiz;
import com.cmj.oa.biz.DepartmentBiz;
import com.cmj.oa.biz.EmployeeBiz;
import com.cmj.oa.entity.ClaimVoucher;
import com.cmj.oa.entity.Department;
import com.cmj.oa.entity.Employee;
import com.cmj.oa.global.Contant;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Controller("employeeController")
@RequestMapping("/employee")
public class EmployeeController {
    @Autowired
    private DepartmentBiz departmentBiz;
    @Autowired
    private EmployeeBiz employeeBiz;
    @Autowired
    private ClaimVoucherBiz claimVoucherBiz;
    @GetMapping("/list")
    public String list(Map<String,Object> map){
        map.put("list",employeeBiz.getAll());
        return "employee_list";
    }

    @RequestMapping("/to_add")
    public String toAdd(Employee employee,Map<String,Object> map){
        map.put("employee",employee);
        map.put("dlist",departmentBiz.getdlist());
        map.put("plist",Contant.getPost());
        return "employee_add";
    }

    @PostMapping("/add")
    public String add(Employee employee){
        employeeBiz.add(employee);
        return "redirect:list";
    }

    @RequestMapping(value = "/to_update",params = "sn")
    public String toUpdate(String sn, Map<String,Object> map){
        map.put("employee",employeeBiz.get(sn));
        map.put("dlist",departmentBiz.getdlist());
        map.put("plist",Contant.getPost());
        return "employee_update";
    }

    @RequestMapping("/update")
    public String update(Employee employee){
        employeeBiz.edit(employee);
        return "redirect:list";
    }

    @RequestMapping(value = "/remove",params = "sn")
    public String remove(String sn){
        //根据员工工号找到对应报销单 再根据对应的报销单id删除报销单 最后删除员工信息
        List<ClaimVoucher> claimVoucherList = claimVoucherBiz.getForSelf(sn);
        if (!claimVoucherList.isEmpty() && claimVoucherList != null){
            List<Integer> ids = new ArrayList<>();
            for (ClaimVoucher claimVoucher : claimVoucherList){
                ids.add(claimVoucher.getId());
            }
            for (int id : ids){
                claimVoucherBiz.deleteById(id);
            }
        }
        employeeBiz.remove(sn);
        return "redirect:list";
    }


}
