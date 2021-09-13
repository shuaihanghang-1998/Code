package com.cmj.oa.dao;

import com.cmj.oa.entity.ClaimVoucherItem;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository("claimVoucherItemDao")
public interface ClaimVoucherItemDao {

    void insert(ClaimVoucherItem claimVoucherItem);
    void update(ClaimVoucherItem claimVoucherItem);
    void delete(int id);
    void deleteByClaimVoucher(int id);
    List<ClaimVoucherItem> selectByClaimVoucher(int cvid);
}
