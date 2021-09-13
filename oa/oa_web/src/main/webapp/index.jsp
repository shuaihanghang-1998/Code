<%--<%@ page contentType="text/html;charset=UTF-8" language="java" %>--%>

<%--<!DOCTYPE html>--%>
<%--<html>--%>
<%--<head>--%>
<%--    <script>--%>
<%--        window.location.href = "./to_login";--%>
<%--    </script>--%>
<%--</head>--%>
<%--<body>--%>
<%--</body>--%>
<%--</html>--%>

<%
    request.getRequestDispatcher("./WEB-INF/pages/login.jsp").forward(request,response);
%>