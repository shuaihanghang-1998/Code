<%@ taglib prefix="spring" uri="http://www.springframework.org/tags" %>
<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core" %>
<%@page import="com.cmj.oa.global.Contant" %>
<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<jsp:include page="top.jsp"/>
<!--添加 layui  支持加载-->
<link rel="stylesheet"	href="/static/layui-v2.4.5/layui/css/layui.css">
<script src="/static/layui-v2.4.5/layui/layui.js" charset="UTF-8"></script>

<script type="text/javascript" src="/javascript/click.js"></script>

<section id="content" class="table-layout animated fadeIn">
    <div class="tray tray-center">
        <div class="content-header">
            <h2> 待处理报销单 </h2>
            <p class="lead"></p>
        </div>
        <div class="admin-form theme-primary mw1200 center-block" style="padding-bottom: 175px;">
            <div class="panel  heading-border">
                <div class="panel-menu">
                    <div class="row">
                        <div class="hidden-xs hidden-sm col-md-2">
                            <div class="btn-group">
                                <button type="button" class="btn btn-default light" onclick="refreshForDeal()">
                                    <i class="fa fa-refresh"></i>
                                </button>
                                <button type="button" class="btn btn-default light" onclick="exportExcel()">
                                    <i class="fa fa-download"></i>
                                </button>
                            </div>
                        </div>
                        <div name="selectFor" class="col-md-8 row">
                            <form action="/claim_voucher/fuzzyQueryForDeal" style="display: inline" method="post" autocomplete="off">
                                <button type="submit" class="btn btn-default light">
                                    <i class="fa fa-search"></i>
                                </button>
                                <button type="reset" class="btn btn-default light">
                                    <i class="fa fa-eraser"></i>
                                </button>
                                <label for="status"  class="field prepend-icon mw100">
                                    <label for="status"  class="field-icon">
                                        <i class="fa fa-reorder"></i>
                                    </label>
                                    <select name="status" id="status" class="gui-input" style="padding-left: 30%">
                                        <option value="" hidden>状态...</option>
                                        <option value="新创建">新创建</option>
                                        <option value="已提交">已提交</option>
                                        <option value="待复审">待复审</option>
                                        <option value="已打回">已打回</option>
                                    </select>
                                </label>
                                <label for="createSn" class="field prepend-icon mw140">
                                    <label for="createSn"  class="field-icon">
                                        <i class="fa fa-user"></i>
                                    </label>
                                    <input type="text" id="createSn" name="createSn" class="gui-input"  placeholder="创建人..." >
                                </label>
                                <label for="total_amount"  class="field prepend-icon mw140">
                                    <label for="total_amount"  class="field-icon">
                                        <i class="fa fa-jpy"></i>
                                    </label>
                                    <input type="text" id="total_amount" name="total_amount" class="gui-input"  placeholder="金额..." >
                                </label>
                                <label for="create_time"  class="field prepend-icon mw140">
                                    <label for="create_time"  class="field-icon">
                                        <i class="fa fa-clock-o"></i>
                                    </label>
                                    <input type="text" name="create_time" id="create_time" class="gui-input"  placeholder="创建时间..." >
                                </label>
                            </form>
                        </div>
                        <div class="col-xs-12 col-md-2 text-right">
                            <div class="btn-group">
                                <button type="button" class="btn btn-default light">
                                    <i class="fa fa-chevron-left"></i>
                                </button>
                                <button type="button" class="btn btn-default light">
                                    <i class="fa fa-chevron-right"></i>
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="panel-body pn">
                    <table id="message-table" class="table admin-form theme-warning tc-checkbox-1" style="text-align: center;">
                        <thead>
                        <tr class="">
                            <th class="text-center hidden-xs">
                                <input class="checkbox" type="checkbox" id="selectAll" value="0" onclick='check_all();'/>
                                Select
                            </th>
                            <th class="text-center hidden-xs">报销类型</th>
                            <th class="text-center hidden-xs">状态</th>
                            <th class="text-center hidden-xs">创建人</th>
                            <th class="text-center hidden-xs">金额</th>
                            <th class="text-center hidden-xs">创建时间</th>
                            <th class="text-center hidden-xs">操作</th>
                        </tr>
                        </thead>
                        <tbody>
                        <c:forEach items="${list}" var="cv">
                            <tr class="message-unread">
                                <td class="hidden-xs">
                                    <label class="hidden-xs">
                                        <input class="checkbox" type="checkbox" name="check" onclick="toCheck()" value="${cv.id}">
                                    </label>
                                </td>
                                <td>${cv.cause}</td>
                                <td class="hidden-xs">
                                    <span class="badge badge-warning fs11">${cv.status}</span>
                                </td>
                                <td>${cv.creater.name}</td>
                                <td class="text-center fw600">${cv.totalAmount}</td>
                                <td><spring:eval expression="cv.createTime"/></td>
                                <td>
                                    <c:if test="${cv.status==Contant.CLAIMVOCHER_CREATED || cv.status==Contant.CLAIMVOCHER_BACK}">
                                        <a style="color: #00acee" href="/claim_voucher/to_update?id=${cv.id}">修改</a>
                                        <a style="color: #00acee" href="/claim_voucher/submit?id=${cv.id}">提交</a>
                                    </c:if>
                                    <c:if test="${cv.status == Contant.CLAIMVOCHER_SUBMIT || cv.status == Contant.CLAIMVOCHER_RECHECK}">
                                        <a style="color: #00acee" href="/claim_voucher/to_check?id=${cv.id}">审核</a>
                                    </c:if>
                                    <c:if test="${cv.status == Contant.CLAIMVOCHER_APPROVED}">
                                        <a style="color: #00acee" href="/claim_voucher/to_check?id=${cv.id}">打款</a>
                                    </c:if>
                                    <a style="color: #00acee" href="/claim_voucher/detail?id=${cv.id}">详细信息</a>
                                </td>
                            </tr>
                        </c:forEach>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
</section>

<jsp:include page="bottom.jsp"/>
