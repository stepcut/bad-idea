{-# language RecordWildCards #-}
module Main where

-- https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

import Data.Vector ((!), Vector)
import qualified Data.Vector as Vector

-- | sigmoid function
sigmoid :: Float -> Float
sigmoid t = 1/(1 + exp (-t))

sigmoid' :: Float -> Float
sigmoid' z = (sigmoid z) * (1 - sigmoid z)


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
      -- we want dE_total / dw5
      -- via the chain rule that is
      -- dE_total / dw5 = dE_total / dout_o11 * dout_o1 / dnet_o1 * dnet_o1 / dw5
      dE_douto1 = -(targeto1 - outo1)
      douto1_dneto1 = (outo1 * (1 - outo1))
      dW5 = dE_douto1 * douto1_dneto1 * outh1
      -- for dE_total / dW6
      -- we have almost the same thing, except dW6 is the weight for outh2 instead of outh1
      dW6 = dE_douto1 * douto1_dneto1 * outh2

      dE_douto2 = -(targeto2 - outo2)
      douto2_dneto2 = (outo2 * (1 - outo2))
      dW7 = dE_douto2 * douto2_dneto2  * outh1
      dW8 = dE_douto2 * douto2_dneto2  * outh2
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
