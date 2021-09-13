function refresh() {
    location.reload();
}

function refreshForSelf(){
    location.href='/claim_voucher/self';
}

function refreshForDeal(){
    location.href='/claim_voucher/deal';
}

layui.use("layer",function(){
    var layer = layui.layer;
});

function remove() {
    layer.open({
        title: false,
        closeBtn: false,
        btnAlign: 'c',
        anim: 6,
        icon: 3,
        textAlign: 'c',
        btn: ['Yes', 'No'],
        btn1: function(index){
            var checkboxs = document.getElementsByName("check");
            var id = new Array();
            for (var i =0; i < checkboxs.length; i++){
                if (checkboxs[i].checked){
                    id.push(checkboxs[i].value);
                }
            }
            if (id.length==0){
                layer.msg('未选择报销单，请选择',{icon: 7});
                return false;
            }else {
                id = id.toString();
            }
            $.ajax({
                type : 'POST',
                url: '/claim_voucher/delete',
                async:false,
                data: {ids:id},
                success : function(result) {
                    layer.msg('删除成功',{icon: 1,time: 2000});
                    location.reload();
                },
                error:function(){
                    layer.msg('服务器错误',{icon: 7});
                }
            });
            layer.close(index);
        },
        btn2: function(index){
            layer.close(index);
        },
        content: '确认删除所选报销单吗？'
    });
}

function check_all() {
    var selectAll = document.getElementById("selectAll");
    var flag = selectAll.checked;
    var checkboxs = document.getElementsByName("check");
    if (flag){
        for (var i = 0; i <checkboxs.length; i++){
            checkboxs[i].checked = true;
        }
    }else {
        for (var i =0; i < checkboxs.length; i++){
            checkboxs[i].checked = false;
        }
    }
}

function toCheck() {
    var selectAll = document.getElementById("selectAll");
    var checkboxs = document.getElementsByName("check");
    count = 0;
    for (var i =0; i < checkboxs.length; i++){
        if (checkboxs[i].checked){
            count++;
        }
    }
    if (count == checkboxs.length){
        selectAll.checked = true;
    }else {
        selectAll.checked = false;
    }
}

function exportExcel() {
    var checkboxs = document.getElementsByName("check");
    var id = new Array();
    for (var i =0; i < checkboxs.length; i++){
        if (checkboxs[i].checked){
            id.push(checkboxs[i].value);
        }
    }
    if (id.length==0){
        layer.msg('未选择报销单，请选择',{icon: 7, time: 2000});
        return false;
    }
    location.href = "/claim_voucher/export?ids="+id.toString();
}

function selectFile() {
    $("#file").trigger("click");
}

function uploadFile(file){
    layer.msg('正在导入，请稍候',{icon: 16,time:false,shade:0.8});
    $.ajaxFileUpload({
        url : '/claim_voucher/upload_excel', //用于文件上传的服务器端请求地址
        secureuri : false, //一般设置为false
        fileElementId : 'file', //文件上传空间的id属性  <input type="file" id="file" name="file" />
        type : 'post',
        dataType : 'text', //返回值类型 一般设置为
        success : function(result) //服务器成功响应处理函数
        {
            var result = $.parseJSON(result.replace(/<.*?>/ig,""));
            layer.msg(result.msg,{icon: 1});
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
layui.use('laydate', function(){
    var laydate = layui.laydate;

    //执行一个laydate实例
    laydate.render({
        elem: '#create_time', //指定元素
        trigger: 'click'
    });
});