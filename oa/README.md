# 公费医疗报销系统

**IDEA+SSM**报销单管理系统

## oa_biz 事务文件

     |——src
     | |——mian
     | |——Java 源码文件
     | | |——com.cmj.oa.biz 事务接口文件包
     | |  |——impl 接口实现包
     | |  | |——ClaimVoucherBizImpl 报销单功能事务接口实现
     | |  | |——DepartmentBizImpl 部门管理功能事务接口实现
     | |  | |——EmployeeBizImpl 员工管理功能事务接口实现
     | |  | |——GlobalBizImpl 登录功能事务接口实现
     | |  |——ClaimVoucherBiz 报销单功能事务接口
     | |  |——DepartmentBiz 部门管理功能事务接口
     | |  |——EmployeeBiz 员工管理功能事务接口
     | |  |——GlobalBiz 登录功能事务接口
     | |——resource 资源文件
     | | |——spring-biz.xml spring事务配置文件

## oa_dao 数据文件

     |——src
     | |——mian
     | |——Java 源码文件
     | | |——com.cmj.oa
     | | | |——dao 数据操作接口包
     | | | | |——ClaimVoucherDao 报销单数据操作接口
     | | | | |——ClaimVoucherItemDao 报销单管理数据操作接口
     | | | | |——DealRecordDao 报销单处理流程数据操作接口
     | | | | |——DepartmentDao 公司部门的数据操作接口
     | | | | |——EmployeeDao 公司员工数据操作接口
     | | | |——entity 数据实体包
     | | | | |——ClaimVoucher 报销单数据实体
     | | | | |——ClaimVoucherItem 报销单管理数据实体
     | | | | |——DealRecord 报销单处理流程数据实体
     | | | | |——Department 公司部门数据实体
     | | | | |——Employee 公司员工数据实体
     | | | |——global 常量文件包
     | | | | |——Contant 常量文件数据
     | |——resource 资源文件
     | | |——com.cmj.oa.dao 资源数据的配置文件包
     | | | |——ClaimVoucherDao.xml 报销单数据 数据库操作配置映射文件
     | | | |——ClaimVoucherItemDao.xml 报销单管理 数据库操作配置映射文件
     | | | |——DealRecordDao.xml 报销单处理流程 数据库操作配置映射文件
     | | | |——DepartmentDao.xml 公司部门管理 数据库操作配置映射文件
     | | | |——EmployeeDao.xml 公司员工管理 数据库操作配置映射文件
     | | |——spring-dao.xml 数据库链接配置，spring-mybatis配置文件

## oa_web网页文件

     |——src
     | |——mian
     | |——Java 源码文件
     | | |——com.cmj.oa
     | | | |——controller 控制器文件
     | | | | |——ClaimVoucherController 报销单功能控制器
     | | | | |——DepartmentController 公司部门功能控制器
     | | | | |——EmployeeController 公司员工功能控制器
     | | | | |——GlobalController 登录功能控制器
     | | | |——dto 数据集合文件
     | | | | |——ClaimVoucherInfo 报销单数据和报销单管理的集合
     | | | |——global
     | | | | |——EncodingFilter 编码格式过滤器
     | | | | |——LoginInterceptor 登录状态检测和拦截
     | |——resource 资源文件
     | | |——spring-web.xml spring-web配置文件
     | |——webapp
     | | |——assets assets插件文件包
     | | |——javaScript js文件包
     | | |——static 
     | | | |——excel 存放导入导出的表格
     | | | |——layui 框架
     | | |——vendor vendor插件文件包
     | | |——WEB-INF
     | | | |——pages jsp文件管理包
     | | | | |——bottom.jsp 底部模板
     | | | | |——change_password.jsp 修改密码页面
     | | | | |——claim_voucher_add.jsp 填写报销单页面
     | | | | |——claim_voucher_check.jsp 处理报销单页面
     | | | | |——claim_voucher_deal.jsp 待处理报销单页面
     | | | | |——claim_voucher_detail.jsp 报销单详细信息页面
     | | | | |——claim_voucher_self.jsp 个人报销单信息页面
     | | | | |——claim_voucher_update.jsp 报销单的修改信息页面
     | | | | |——department_add.jsp 添加部门信息页面
     | | | | |——department_list.jsp 部门信息列表页面
     | | | | |——department_update.jsp 修改部门信息页面
     | | | | |——employee_add.jsp 添加员工信息页面
     | | | | |——employee_list.jsp 员工信息列表页面
     | | | | |——employee_update.jsp 修改员工信息页面
     | | | | |——login.jsp 登录页面
     | | | | |——self.jsp 个人信息页面
     | | | | |——top.jsp 顶部模板
     | | | |——web.xml web配置文件
     | | |——index.jsp