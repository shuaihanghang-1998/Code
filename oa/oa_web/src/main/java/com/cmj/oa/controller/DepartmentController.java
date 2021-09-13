package com.cmj.oa.controller;

import com.cmj.oa.biz.ClaimVoucherBiz;
import com.cmj.oa.biz.DepartmentBiz;
import com.cmj.oa.biz.EmployeeBiz;
import com.cmj.oa.dao.EmployeeDao;
import com.cmj.oa.entity.ClaimVoucher;
import com.cmj.oa.entity.Department;
import com.cmj.oa.entity.Employee;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Controller("departmentController")
@RequestMapping("/department")
public class DepartmentController {
    @Autowired
    private DepartmentBiz departmentBiz;
    @Autowired
    private ClaimVoucherBiz claimVoucherBiz;
    @Autowired
    private EmployeeBiz employeeBiz;
    @Qualifier("employeeDao")
    @Autowired
    private EmployeeDao employeeDao;
    @RequestMapping("/list")
    public String list(Map<String,Object> map){
        map.put("list",departmentBiz.getAll());
        return "department_list";
    }

    @RequestMapping("/to_add")
    public String toAdd(Map<String,Object> map){
        map.put("department",new Department());
        return "department_add";
    }

    @RequestMapping("/add")
    public String add(Department department){
        departmentBiz.add(department);
        return "redirect:list";
    }

    @RequestMapping(value = "/to_update",params = "sn")
    public String toUpdate(String sn, Map<String,Object> map){
        map.put("department",departmentBiz.get(sn));
        return "department_update";
    }

    @RequestMapping("/update")
    public String update(Department department){
        departmentBiz.edit(department);
        return "redirect:list";
    }

    @RequestMapping(value = "/remove",params = "sn")
    public String remove(String sn){
        //根据部门号找到对应的员工
        List<Employee> employeeList = employeeDao.selectByDepartmentAndPost(sn,null);
        List<ClaimVoucher> claimVoucherList = null;
        List<Integer> ids = null;
        if (!employeeList.isEmpty() && employeeList != null){
            //删除员工及其报销单
            for (Employee employee : employeeList){
                //根据报销单id删除报销单
                claimVoucherList = claimVoucherBiz.getForSelf(employee.getSn());
                if (!claimVoucherList.isEmpty() && claimVoucherList != null){
                    ids = new ArrayList<>();
                    for (ClaimVoucher claimVoucher : claimVoucherList){
                        ids.add(claimVoucher.getId());
                    }
                    for (int id : ids){
                        claimVoucherBiz.deleteById(id);
                    }
                }
                employeeBiz.remove(employee.getSn());
            }
        }
        departmentBiz.remove(sn);
        return "redirect:list";
    }


}
