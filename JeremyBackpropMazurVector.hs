{-# language FlexibleContexts, GADTs, RecordWildCards, TypeSynonymInstances, FlexibleInstances, TypeOperators #-}
module Main where

-- https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

-- https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c

-- import Data.Vector (Vector)
import Data.Array.Accelerate (All(..), Acc, Exp, Elt, Vector, Matrix, Z(..), (:.)((:.)), fromList, lift, unlift, use, (+), (/), zipWith, map, unit, fold, shape)
import qualified Data.Array.Accelerate as A

-- import Data.Array.Accelerate.Linear.Matrix ((!*))
-- import Data.Array.Accelerate.Linear.Vector ((^+^))
-- import Data.Array.Accelerate.Linear.Vector (Additive(..))
-- import Data.Array.Accelerate.Interpreter (run)
import Data.Array.Accelerate.LLVM.Native (run)
import Debug.Trace (trace)
import Prelude hiding ((+), (/), zipWith, map)
{-
import Data.Vector ((!), Vector, snoc)
import qualified Data.Vector as Vector
import Linear.Algebra (mult)
import Linear.Metric (dot)
import Linear.Vector (sumV, (^+^), (^-^), (^*), (*^),(^/))
import Linear.Matrix (transpose, (!*))
-}

-- instance Additive Vector

{-
-- | sigmoid function
sigmoid :: Float -> Float
sigmoid t = 1/(1 + exp (-t))
-}

sigmoid_f :: Float -> Float
sigmoid_f t = 1/(1 + exp (-t))

sigmoid :: (Floating (Exp a), Elt a) => Exp a -> Exp a
sigmoid t = 1 / (1 + A.exp (A.negate t))

{-
sigmoid' :: Float -> Float
sigmoid' z = (sigmoid z) * (1 - sigmoid z)
-}

{-

This is a matrix that has all the weights from one layer to another. With out matrices we might have:

     w1
i1 ------> j1
    \   / w2
     \ /
      x
     / \
    /   \ w3
i2-/-----\-> j2
     w4

With a matrix we have:

     w(0,0)
i1 ------> j1
    \   / w(0,1)
     \ /
      x
     / \
    /   \ w(1,0)
i2-/-----\-> j2
     w(1,1)


So w1 = (0,0)
   w2 = (1,0)
   w3 = (0,1)
   w4 = (1,1)

Or:

  w_ji

        j
      | 0  1
    -----------
i = 0 | w1 w2
    1 | w3 w4

-}
type Layer = Matrix Double

data Network = Network
  { layers :: [Layer]
  , biases :: [Double]
  }
  deriving (Eq, Show)

inp :: Vector Double
inp = fromList (Z:.2) [0.05, 0.10]

mm :: Matrix Double
mm = fromList (Z:.2:.3) [ 1, 2, 3
                        , 4, 5 ,6
                        ]

vv :: Vector Double
vv = fromList (Z:.3) [ 7, 8, 9 ]

(^+^) :: (Num (Exp a), Elt a) => Acc (Vector a) -> Acc (Vector a) -> Acc (Vector a)
(^+^) xs ys = zipWith (+) xs ys

-- (^-^) :: (Num (Exp a), Elt a) => Acc (Vector a) -> Acc (Vector a) -> Acc (Vector a)
(^-^) :: (A.Shape sh, Num (Exp c), Elt c) => Acc (A.Array sh c) -> Acc (A.Array sh c) -> Acc (A.Array sh c)
(^-^) xs ys = zipWith (-) xs ys

-- (^*^) :: (Num (Exp a), Elt a) => Acc (Vector a) -> Acc (Vector a) -> Acc (Vector a)
(^*^) :: (A.Shape sh, Num (Exp c), Elt c) => Acc (A.Array sh c) -> Acc (A.Array sh c) -> Acc (A.Array sh c)
(^*^) xs ys = zipWith (*) xs ys

--(!*) :: (Num (Exp a), Elt a) => Acc (Matrix a) -> Acc (Vector a) -> Acc (Vector a)

-- matrix-vector multiplication
(!*) :: (Num (Exp a), Elt a) => Acc (Matrix a) -> Acc (Vector a) -> Acc (Vector a)
mat !* vec =
    let Z :. rows :. cols = unlift (shape mat) :: Z :. Exp Int :. Exp Int
        vec'              = A.replicate (lift (Z :. rows :. All)) vec
    in fold (+) 0 (zipWith (*) mat vec')

-- muliple two vectors to get a matrix -- could be generalized to two matrixes
-- https://en.wikipedia.org/wiki/Matrix_multiplication
-- this is not right -- there should be a sumation?
(!*!) :: (Num (Exp a), Elt a) => Acc (Vector a) -> Acc (Vector a) -> Acc (Matrix a)
v1 !*! v2 =
  let Z :. rows1 = unlift (shape v1) :: Z :. Exp Int
      Z :. rows2 = unlift (shape v2) :: Z :. Exp Int
      v1' = A.replicate (lift (Z :.  All :. rows1 )) v1
      v2' = A.replicate (lift (Z :. rows2 :. All )) v2
  in zipWith (*) v1' v2'
     -- undefined -- zipWith (*) v1 v2

activate :: (Floating (Exp a), Elt a) => Acc (Vector a) -> Acc (Vector a)
activate v = map sigmoid v

data LayerValues = LayerValues
  { net :: Vector Double
  , out :: Vector Double
  }
  deriving Show

data ForwardPass = ForwardPass
  { inputs      :: Vector Double
  , layerValues :: [LayerValues]
  }
  deriving Show

feedForward :: Vector Double -> Network -> ForwardPass
feedForward inputs (Network layers biases) =
    let h_net = (map (\r -> lift (biases!!0) + r) (((use (layers!!0)) !* use inputs)))
        h_net' = run h_net
        h_out = activate $ use h_net'
        o_net = (map (\r -> lift (biases!!1) + r) (((use (layers!!1)) !* h_out)))
        o_net' = run o_net
        o_out = activate $ use o_net'
    in ForwardPass { inputs = inputs
                   , layerValues = [ LayerValues { net = h_net'
                                                 , out = run h_out
                                                 }
                                   , LayerValues { net = o_net'
                                                 , out = run o_out
                                                 }
                                   ]
                   }

backwardsPass :: Network -> ForwardPass -> Vector Double -> Network
backwardsPass n@Network{..} f@ForwardPass{..} target =
  let alpha = 0.5
      -- instead of o1, o2 as separate values we have a vector o
      weights_ho = use (layers!!1) -- weights from h -> o. aka, w5, w6, w7, w8
      out_o = use (out (layerValues!!1))
      out_h = use (out (layerValues!!0))

      dE_dout = out_o ^-^ (use target)

      dout_dnet out = map (\o -> o * (1 - o)) out

      dout_dnet_o = dout_dnet out_o
      dWho = (dE_dout ^*^  dout_dnet_o) !*! out_h
      weights_ho' = weights_ho ^-^ (map (\w -> alpha * w) dWho)
  in trace (" {{{ " ++ (show $ run weights_ho') ++ " }}}") n

initialNetwork :: Network
initialNetwork = Network
  { layers = [ fromList (Z:.2:.2) [ 0.15, 0.20
                                  , 0.25, 0.30
                                  ]
             , fromList (Z:.2:.2) [ 0.40, 0.45
                                  , 0.50, 0.55
                                  ]
             ]
  , biases = [ 0.35, 0.60 ]
  }

main =
   do let forwardPass = feedForward inp initialNetwork
      print forwardPass
      let target = fromList  (Z :. 2) [ 0.01, 0.99 ]
      putStrLn $ "target = " ++ show target
      let back = backwardsPass initialNetwork forwardPass target
      putStrLn $ "back = " ++ show back
      pure ()


{-
-- feedForward :: Vector Double -> Network -> Vector Double
feedForward inputs (Network layers biases) =
  ((layers!!0) !* inputs) ^+^ biases
--   ((layers!!0) `dot` inputs) ^+^ (biases!!0)

initialNetwork :: Network
initialNetwork = Network
  { layers = [ Vector.fromList [ Vector.fromList [ 0.15, 0.20, 0.25, 0.30 ]
                               , Vector.fromList [ 0.40, 0.45, 0.50, 0.55 ]
                               ]
             ]
  , biases = Vector.fromList [ 0.35, 0.60 ]
  }
-}
{-
data Network = Network
 { w1 :: Float
 , w2 :: Float
 , w3 :: Float
 , w4 :: Float
 , b1 :: Float
 , w5 :: Float
 , w6 :: Float
 , w7 :: Float
 , w8 :: Float
 , b2 :: Float
 }
 deriving (Show)


initialNetwork :: Network
initialNetwork =
  Network { w1 = 0.15
          , w2 = 0.20
          , w3 = 0.25
          , w4 = 0.30
          , b1 = 0.35
          , w5 = 0.40
          , w6 = 0.45
          , w7 = 0.50
          , w8 = 0.55
          , b2 = 0.60
          }

data ForwardPass = ForwardPass
 { i1    :: Float
 , i2    :: Float
 , neth1 :: Float
 , outh1 :: Float
 , neth2 :: Float
 , outh2 :: Float
 , neto1 :: Float
 , outo1 :: Float
 , neto2 :: Float
 , outo2 :: Float
 }
 deriving Show

forwardPass :: Network -> (Float, Float) -> ForwardPass
forwardPass (Network {..}) (i1, i2) =
  let neth1 = (w1 * i1) + (w2 * i2) + (b1 * 1)
      outh1 = sigmoid neth1
      neth2 = (w3 * i1) + (w4 * i2) + (b1 * 1)
      outh2 = sigmoid neth2
      neto1 = (w5 * outh1) + (w6 * outh2) + (b2 * 1)
      outo1 = sigmoid neto1
      neto2 = (w7 * outh1) + (w8 * outh2) + (b2 * 1)
      outo2 = sigmoid neto2
  in (ForwardPass i1 i2 neth1 outh1 neth2 outh2 neto1 outo1 neto2 outo2)



eTotal :: (Float, Float) -> (Float, Float) -> Float
eTotal (target_o1, target_o2) (output_o1, output_o2) =
  let e_o1 = 0.5 * (target_o1 - output_o1)^2
      e_o2 = 0.5 * (target_o2 - output_o2)^2
  in e_o1 + e_o2

-- backwardsPass :: Network -> ForwardPass -> (Float, Float) -> Network
backwardsPass n@Network{..} ForwardPass{..} (targeto1, targeto2) =
  let alpha = 0.5
      dE_douto1 = -(targeto1 - outo1)
      douto1_dneto1 = (outo1 * (1 - outo1))
      dW5 = dE_douto1 * douto1_dneto1 * outh1
      dW6 = dE_douto1 * douto1_dneto1 * outh2
      dE_douto2 = -(targeto2 - outo2)
      douto2_dneto2 = (outo2 * (1 - outo2))
      dW7 = dE_douto2 * douto2_dneto2  * outh1
      dW8 = dE_douto2 * douto2_dneto2 * outh2
      dEo1_dneto1 = dE_douto1 * douto1_dneto1
--      dEo1_douth1 = dEo1_dneto1 * w5
      -- hidden layers
      dEo1_douth1 = dE_douto1 * douto1_dneto1 * w5
      dEo2_dneto2 = dE_douto2 * douto2_dneto2
      dEo2_douth1 = dEo2_dneto2 * w7
      dE_douth1 = dEo1_douth1 + dEo2_douth1
      douth1_dneth1 = outh1 * (1 - outh1)
      dE_dw1 = dE_douth1 * douth1_dneth1 * i1
      dE_dw2 = dE_douth1 * douth1_dneth1 * i2

      dEo1_douth2 = dE_douto1 * douto1_dneto1 * w7
      dEo2_douth2 = dE_douto2 * douto2_dneto2 * w8
      dE_outh2 = dEo1_douth2 + dEo2_douth2
      douth2_dneth2 = outh2 * (1 - outh2)
      dE_dw3 = dE_outh2 * douth2_dneth2 * i1
      dE_dw4 = dE_outh2 * douth2_dneth2 * i2

  in n { w1 = w1 - (alpha * dE_dw1)
       , w2 = w2 - (alpha * dE_dw2)
       , w3 = w3 - (alpha * dE_dw3)
       , w4 = w4 - (alpha * dE_dw4)
       , w5 = w5 - (alpha * dW5)
       , w6 = w6 - (alpha * dW6)
       , w7 = w7 - (alpha * dW7)
       , w8 = w8 - (alpha * dW8)
       }

train :: ((Float, Float), (Float, Float)) -> Network -> Network
train (input, target) network =
  let fp = forwardPass network input
  in backwardsPass network fp target

main =
  let fp = (forwardPass initialNetwork  (0.05, 0.10))
      target = (0.01, 0.99)
      e = eTotal target (outo1 fp, outo2 fp)
      bp = backwardsPass initialNetwork fp target
  in print $ let network = (iterate (train ((0.05, 0.10), target)) initialNetwork)!!100000
             in forwardPass network (0.05, 0.10)
-}


