package com.cmj.oa.dao;

import com.cmj.oa.entity.ClaimVoucher;
import org.apache.ibatis.annotations.Param;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository("claimVoucherDao")
public interface ClaimVoucherDao {
    void insert(ClaimVoucher claimVoucher);
    void update(ClaimVoucher claimVoucher);
    void delete(int id);
    ClaimVoucher select(int id);
    List<ClaimVoucher> selectByCreateSn(String csn);
    List<ClaimVoucher> selectByNextDealSn(String ndsn);
    List<ClaimVoucher> fuzzyQuery(@Param("createSn")String createSn,@Param("status")String status, @Param("totalAmount")Double totalAmount, @Param("create_time")String create_time);
    List<ClaimVoucher> fuzzyQueryForDeal(@Param("createSn")String createSn,@Param("nextDealsn")String nextDealsn,@Param("status")String status, @Param("totalAmount")Double totalAmount, @Param("create_time")String create_time);
}
