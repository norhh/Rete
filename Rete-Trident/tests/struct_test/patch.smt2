(declare-const lvalue_x (_ BitVec 32))
(declare-const rvalue_x (_ BitVec 32))
(declare-const rreturn (_ BitVec 32))
(assert (and (= rreturn (_ bv0 32)) (= lvalue_x (bvadd rvalue_x (_ bv1 32)))))


(declare-const lvalue_a[0] (_ BitVec 32))
(declare-const rreturn (_ BitVec 32))
(assert (and (= rreturn (_ bv0 32)) (= lvalue_[0] (_ bv99 32))))