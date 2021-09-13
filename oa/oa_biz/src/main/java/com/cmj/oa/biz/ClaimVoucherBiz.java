package com.cmj.oa.biz;

import com.cmj.oa.entity.ClaimVoucher;
import com.cmj.oa.entity.ClaimVoucherItem;
import com.cmj.oa.entity.DealRecord;
import com.cmj.oa.entity.DateUtil;

import java.util.List;

public interface ClaimVoucherBiz {

    void save(ClaimVoucher claimVoucher, List<ClaimVoucherItem> items);

    void deleteById(int id);

    ClaimVoucher get(int id);

    List<ClaimVoucherItem> getItems(int cvid);

    List<DealRecord> getRecords(int cvid);

    List<ClaimVoucher> getForSelf(String sn);

    List<ClaimVoucher> getForDeal(String sn);

    void update(ClaimVoucher claimVoucher,List<ClaimVoucherItem> items);

    void submit(int id);

    void deal(DealRecord dealRecord);

    List<ClaimVoucher> fuzzyQuery(String createSn,String status,Double totalAmount,String create_time);

    List<ClaimVoucher> fuzzyQueryForDeal(String createSn,String nextDealsn,String status,Double totalAmount,String create_time);

}
