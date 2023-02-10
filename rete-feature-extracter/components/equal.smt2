(declare-const rhole_left (_ BitVec 32))
(declare-const rhole_right (_ BitVec 32))
(declare-const rreturn (_ BitVec 32))
(assert  ( ite (= rhole_left rhole_right) (= rreturn (_ bv1 32) ) (= rreturn (_ bv0 32 ))))
