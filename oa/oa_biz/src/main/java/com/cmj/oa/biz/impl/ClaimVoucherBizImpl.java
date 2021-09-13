package com.cmj.oa.biz.impl;

import com.cmj.oa.biz.ClaimVoucherBiz;
import com.cmj.oa.dao.ClaimVoucherDao;
import com.cmj.oa.dao.ClaimVoucherItemDao;
import com.cmj.oa.dao.DealRecordDao;
import com.cmj.oa.dao.EmployeeDao;
import com.cmj.oa.entity.ClaimVoucher;
import com.cmj.oa.entity.ClaimVoucherItem;
import com.cmj.oa.entity.DealRecord;
import com.cmj.oa.entity.Employee;
import com.cmj.oa.global.Contant;
import org.apache.commons.fileupload.FileItem;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Service;

import java.util.Date;
import java.util.List;
import com.cmj.oa.entity.DateUtil;

@Service("claimVoucherBiz")
public class ClaimVoucherBizImpl implements ClaimVoucherBiz {

    @Qualifier("claimVoucherDao")
    @Autowired
    private ClaimVoucherDao claimVoucherDao;
    @Qualifier("claimVoucherItemDao")
    @Autowired
    private ClaimVoucherItemDao claimVoucherItemDao;
    @Qualifier("dealRecordDao")
    @Autowired
    private DealRecordDao dealRecordDao;
    @Qualifier("employeeDao")
    @Autowired
    private EmployeeDao employeeDao;

    @Override
    public void save(ClaimVoucher claimVoucher, List<ClaimVoucherItem> items) {
        claimVoucher.setCreateTime(new Date());
        claimVoucher.setNextDealSn(claimVoucher.getCreateSn());
        claimVoucher.setStatus(Contant.CLAIMVOCHER_CREATED);
        claimVoucherDao.insert(claimVoucher);

        for (ClaimVoucherItem item:items){
//            item.setClaimVoucherId(claimVoucher.getId());
//            item.setComment(DateUtil.formatDate(new Date(), "yyyyMMddHH")+"_"+claimVoucher.getCreateSn()+"_"+item.getComment()+".jpg");
//            claimVoucherItemDao.insert(item);
            item.setClaimVoucherId(claimVoucher.getId());
            String str=item.getComment();
            String filename;
            int pos = str.lastIndexOf("\\");
            filename=str.substring(pos+1);
            item.setComment(DateUtil.formatDate(new Date(), "yyyyMMddHH")+"_"+claimVoucher.getCreateSn()+"_"+filename);
            claimVoucherItemDao.insert(item);
        }
    }

    @Override
    public void deleteById(int id) {
        dealRecordDao.deleteByClaimVoucher(id);
        claimVoucherItemDao.deleteByClaimVoucher(id);
        claimVoucherDao.delete(id);
    }

    @Override
    public ClaimVoucher get(int id) {
        return claimVoucherDao.select(id);
    }

    @Override
    public List<ClaimVoucherItem> getItems(int cvid) {
        return claimVoucherItemDao.selectByClaimVoucher(cvid);
    }

    @Override
    public List<DealRecord> getRecords(int cvid) {
        return dealRecordDao.selectByClaimVoucher(cvid);
    }

    @Override
    public List<ClaimVoucher> getForSelf(String sn) {
        return claimVoucherDao.selectByCreateSn(sn);
    }

    @Override
    public List<ClaimVoucher> getForDeal(String sn) {
        return claimVoucherDao.selectByNextDealSn(sn);
    }

    @Override
    public void update(ClaimVoucher claimVoucher, List<ClaimVoucherItem> items) {
        claimVoucher.setNextDealSn(claimVoucher.getCreateSn());
        claimVoucher.setStatus(Contant.CLAIMVOCHER_CREATED);
        claimVoucherDao.update(claimVoucher);

        List<ClaimVoucherItem> olds = claimVoucherItemDao.selectByClaimVoucher(claimVoucher.getId());
        for (ClaimVoucherItem old:olds){
            boolean isHave = false;
            for (ClaimVoucherItem item:items){
                if(item.getId() == old.getId()){
                    isHave = true;
                    break;
                }
            }
            if(!isHave){
                claimVoucherDao.delete(old.getId());
            }
        }
        for (ClaimVoucherItem item:items){
            item.setClaimVoucherId(claimVoucher.getId());
            String str=item.getComment();
            if(str.charAt(10)=='_'){

            } else{
                String filename;
                int pos = str.lastIndexOf("\\");
                filename=str.substring(pos+1);
                item.setComment(DateUtil.formatDate(new Date(), "yyyyMMddHH")+"_"+claimVoucher.getCreateSn()+"_"+filename);
            }
            if (item.getId()!=null && item.getId() > 0){
                claimVoucherItemDao.update(item);
            }else{
                claimVoucherItemDao.insert(item);
            }
        }
    }

    @Override
    public void submit(int id) {

        ClaimVoucher claimVoucher = claimVoucherDao.select(id);
        Employee employee = employeeDao.select(claimVoucher.getCreateSn());

        claimVoucher.setStatus(Contant.CLAIMVOCHER_SUBMIT);

        if (!employeeDao.selectByDepartmentAndPost(employee.getDepartmentSn(),Contant.POST_FM).isEmpty()) {
            claimVoucher.setNextDealSn(employeeDao.selectByDepartmentAndPost(employee.getDepartmentSn(), Contant.POST_FM).get(0).getSn());
        }
        claimVoucherDao.update(claimVoucher);

        DealRecord dealRecord = new DealRecord();
        dealRecord.setDealWay(Contant.DEAL_SUBMIT);
        dealRecord.setDealSn(employee.getSn());
        dealRecord.setClaimVoucherId(id);
        dealRecord.setDealResult(Contant.DEAL_SUBMIT);
        dealRecord.setDealTime(new Date());
        dealRecord.setComment("无");
        dealRecordDao.insert(dealRecord);

    }

    @Override
    public void deal(DealRecord dealRecord) {
        ClaimVoucher claimVoucher = claimVoucherDao.select(dealRecord.getClaimVoucherId());
        Employee employee = employeeDao.select(dealRecord.getDealSn());
        dealRecord.setDealTime(new Date());

        if (dealRecord.getDealWay().equals(Contant.DEAL_PASS)){
            if(claimVoucher.getTotalAmount() <=  Contant.LIMIT_CHECK || employee.getPost().equals(Contant.POST_GM) ) {
                claimVoucher.setStatus(Contant.CLAIMVOCHER_APPROVED);
                claimVoucher.setNextDealSn(employeeDao.selectByDepartmentAndPost(null, Contant.POST_CAHIER).get(0).getSn());

                dealRecord.setDealResult(Contant.CLAIMVOCHER_APPROVED);
            }else {
                claimVoucher.setStatus(Contant.CLAIMVOCHER_RECHECK);
                claimVoucher.setNextDealSn(employeeDao.selectByDepartmentAndPost(null, Contant.POST_GM).get(0).getSn());

                dealRecord.setDealResult(Contant.CLAIMVOCHER_APPROVED);
            }
        }else if (dealRecord.getDealWay().equals(Contant.DEAL_BACK)){
            claimVoucher.setStatus(Contant.CLAIMVOCHER_BACK);
            claimVoucher.setNextDealSn(claimVoucher.getCreateSn());

            dealRecord.setDealResult(Contant.CLAIMVOCHER_BACK);
        }else if (dealRecord.getDealWay().equals(Contant.DEAL_REJECT)){
            claimVoucher.setStatus(Contant.CLAIMVOCHER_TERMINATED);
            claimVoucher.setNextDealSn(null);

            dealRecord.setDealResult(Contant.CLAIMVOCHER_TERMINATED);
        }else if (dealRecord.getDealWay().equals(Contant.DEAL_PAID)){
            claimVoucher.setStatus(Contant.CLAIMVOCHER_PAID);
            claimVoucher.setNextDealSn(null);

            dealRecord.setDealResult(Contant.CLAIMVOCHER_PAID);
        }

        claimVoucherDao.update(claimVoucher);
        dealRecordDao.insert(dealRecord);
    }

    @Override
    public List<ClaimVoucher> fuzzyQuery(String createSn, String status, Double totalAmount, String create_time) {
        return claimVoucherDao.fuzzyQuery(createSn,status,totalAmount,create_time);
    }

    @Override
    public List<ClaimVoucher> fuzzyQueryForDeal(String createSn, String nextDealsn, String status, Double totalAmount, String create_time) {
        return claimVoucherDao.fuzzyQueryForDeal(createSn,nextDealsn,status,totalAmount,create_time);
    }
}
