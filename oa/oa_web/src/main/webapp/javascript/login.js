window.onload = function(){
    var oUser = document.getElementById('sn');
    var oPswd = document.getElementById('password');
    var oRemember = document.getElementById('remember');
    //页面初始化时，如果帐号密码cookie存在则填充
    if(getCookie('user') && getCookie('pswd')){
        oUser.value = getCookie('user');
        oPswd.value = getCookie('pswd');
        oRemember.checked = true;
    }
    //复选框勾选状态发生改变时，如果未勾选则清除cookie
    oRemember.onchange = function(){
        if(!this.checked){
            delCookie('user');
            delCookie('pswd');
        }
    };
    //表单提交事件触发时，如果复选框是勾选状态则保存cookie
    // oForm.onsubmit = function(){
    //     if($("#remember").is(':checked')){
    //         setCookie('user',oUser.value,7); //保存帐号到cookie，有效期7天
    //         setCookie('pswd',oPswd.value,7); //保存密码到cookie，有效期7天
    //     }
    // };
};

//设置cookie
function setCookie(name,value,day){
    var date = new Date();
    date.setDate(date.getDate() + day);
    document.cookie = name + '=' + value + ';expires='+ date;
};
//获取cookie
function getCookie(name){
    var reg = RegExp(name+'=([^;]+)');
    var arr = document.cookie.match(reg);
    if(arr){
        return arr[1];
    }else{
        return '';
    }
};
//删除cookie
function delCookie(name){
    setCookie(name,null,-1);
};


layui.use("layer",function(){
    var layer = layui.layer;
});

function login() {
    var sn = $("#sn").val();
    var password = $("#password").val();
    if (sn==""){
        layer.msg('工号不能为空，请输入工号',{icon:7,time:1500});
        return false;
    }
    if (password==""){
        layer.msg('密码不能为空，请输入密码',{icon:7,time:1800});
        return false;
    }

    $.ajax({
        type : 'post',
        url :  '/login',
        data : {
            'username' : sn,
            'password' : password
        },
        dataType : 'text',
        success: function(data) {
            if (data === 'true') {
                if($("#remember").is(':checked')){
                    setCookie('user',sn,7); //保存工号到cookie，有效期7天
                    setCookie('pswd',password,7); //保存密码到cookie，有效期7天
                }
                window.location.href = "/self";
            }else {
                layer.msg('工号或密码错误',{icon:2,time:1800});
            }
        }
    })

}