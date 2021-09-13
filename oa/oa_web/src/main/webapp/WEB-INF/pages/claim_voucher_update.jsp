<%@ taglib prefix="form" uri="http://www.springframework.org/tags/form" %>
<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core" %>
<%@ page contentType="text/html;charset=UTF-8" language="java" %>

<jsp:include page="top.jsp"/>

<link rel="stylesheet"	href="/static/layui-v2.4.5/layui/css/layui.css">
<script src="/static/layui-v2.4.5/layui/layui.js" charset="UTF-8"></script>

<section id="content" class="table-layout animated fadeIn">
    <div class="tray tray-center">
        <div class="content-header">
            <h2> 修改报销单 </h2>
            <p class="lead"></p>
        </div>
        <div class="admin-form theme-primary mw1000 center-block" style="padding-bottom: 175px;">
            <div class="panel heading-border">
                <form:form id="admin-form" name="addForm" action="/claim_voucher/update" modelAttribute="info">
                    <form:hidden path="claimVoucher.id"/>
                    <div class="panel-body bg-light">
                        <div class="section-divider mt20 mb40">
                            <span> 基本信息 </span>
                        </div>
                        <div class="section">
                            <label for="claimVoucher.cause" class="field prepend-icon">
                                <form:select id="cause" cssStyle="padding-left: 4%" path="claimVoucher.cause" cssClass="gui-input" placeholder="报销类别..." items="${cause}"/>
<%--                                <form:input path="claimVoucher.cause" cssClass="gui-input" placeholder="事由..."/>--%>
                                <label for="claimVoucher.cause" class="field-icon">
                                    <i class="fa fa-pencil-square"></i>
                                </label>
                            </label>
                        </div>
                        <div class="section-divider mt20 mb40">
                            <span> 费用明细 </span>
                        </div>
                        <div id="items">
                            <c:forEach items="${info.items}" varStatus="sta">
                                <div class="section row">
                                    <div class="col-md-3">
                                        <label for="items[${sta.index}].item" class="field prepend-icon">
                                            <form:hidden path="items[${sta.index}].id"/>
                                            <form:hidden path="items[${sta.index}].claimVoucherId"/>
                                            <label for="items[${sta.index}].item" class="field-icon">
                                                <i class="fa fa-reorder"></i>
                                            </label>
                                            <form:select cssStyle="padding-left: 17%" path="items[${sta.index}].item" cssClass="gui-input" placeholder="花销类型..." items="${items}"/>
                                        </label>
                                    </div>
                                    <div class="col-md-3">
                                        <label for="items[${sta.index}].amount" class="field prepend-icon">
                                            <form:input path="items[${sta.index}].amount" cssClass="gui-input money" placeholder="金额..."/>
                                            <label for="items[${sta.index}].amount" class="field-icon">
                                                <i class="fa fa-jpy"></i>
                                            </label>
                                        </label>
                                    </div>
                                    <div class="col-md-3">
                                        <label for="items[${sta.index}].comment" class="field prepend-icon">
                                            <form:input id="comment" path="items[${sta.index}].comment" cssClass="gui-input" placeholder="备注..." />
                                            <label for="items[${sta.index}].comment" class="field-icon">
                                                <i class="fa fa-edit"></i>
                                            </label>
                                        </label>
                                    </div>
                                    <div class="col-md-2">
                                    <label  class="field prepend-icon">
                                        <input type="file" id="file" name="file" style="display: none" onchange="uploadimage(this)">
                                        <input type="button" class="btn btn-default light" value="修改图片" onclick="selectFile()">
                                        <label for="file" class="field-icon">
                                            <i class="fa fa-upload"></i>
                                        </label>
                                    </label>
                                </div>
                                    <div class="col-md-1" style="text-align:right;">
                                        <button type="button" class="button"> X </button>
                                    </div>
                                </div>
                            </c:forEach>
                        </div>
                        <div class="section row">
                            <div class="col-md-3">
                                <label for="totalMoney" class="field prepend-icon">
                                    <form:input id="totalMoney" path="claimVoucher.totalAmount" cssClass="gui-input" placeholder="总金额..." readonly="true"/>
                                    <label for="totalMoney" class="field-icon">
                                        <i class="fa fa-jpy"></i>
                                    </label>
                                </label>
                            </div>
                            <div class="col-md-3">
                                <label for="totalReimbursementMoney" class="field prepend-icon">
                                    <form:input id="totalReimbursementMoney" path="claimVoucher.totalReimbursementAmount" cssClass="gui-input" placeholder="报销金额..." readonly="true"/>
                                    <label for="totalReimbursementMoney" class="field-icon">
                                        <i class="fa fa-jpy"></i>
                                    </label>
                                </label>
                            </div>
                            <div class="section" style="text-align:right;">
                                <div class="col-md-6">
                                    <button type="button" class="button" id="addItemButton"> + </button>
                                </div>
                            </div>
                        </div>
                        <div class="panel-footer text-right">
                            <button type="submit" class="button"> 保存 </button>
                            <button type="button" class="button" onclick="javascript:window.history.go(-1);"> 返回 </button>
                        </div>
                    </div>
                </form:form>
            </div>
        </div>
    </div>
</section>

<jsp:include page="bottom.jsp"/>
<script src="/javascript/ajaxfileupload.js" charset="UTF-8"></script>

<script type="text/javascript" src="/javascript/click.js" charset="UTF-8"></script>
<script>
    function uploadimage(file){
        layer.msg('正在导入，请稍候',{icon: 16,time:false,shade:0.8});
        $.ajaxFileUpload({
            url : '/claim_voucher/upload_image', //用于文件上传的服务器端请求地址
            secureuri : false, //一般设置为false
            fileElementId : 'file', //文件上传空间的id属性  <input type="file" id="file" name="file" />
            type : 'post',
            dataType : 'text', //返回值类型 一般设置为
            success : function(result) //服务器成功响应处理函数
            {
                var result = $.parseJSON(result.replace(/<.*?>/ig,""));
                layer.msg(result.msg,{icon: 1});
                var str=$("#file").val();
                $("#comment").attr("value",str);
                $("#file").val("");
            },
            error : function(result)//服务器响应失败处理函数
            {
                var result = $.parseJSON(result.replace(/<.*?>/ig,""));
                layer.msg(result.msg,{icon: 2});
                $("#file").val("");
            }
        });
        return false;
    }

</script>