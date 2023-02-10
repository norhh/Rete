(declare-const rhole_left (_ BitVec 32))
(declare-const rhole_right (_ BitVec 32))
(declare-const rreturn (_ BitVec 32))
(assert (= rreturn ( ite (bvslt rhole_left rhole_right) (_ bv1 32) ( _ bv0 32 ))) )
