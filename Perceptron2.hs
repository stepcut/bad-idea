module Main where

import Data.Vector ((!), Vector)
import qualified Data.Vector as Vector
import Debug.Trace (trace)
import Graphics.Gloss hiding (Vector)
import Graphics.Gloss.Interface.Pure.Game hiding (Vector)
import Linear.Algebra (mult)
import Linear.Metric (dot)
import Linear.Vector (sumV, (^+^), (^-^), (^*), (*^),(^/))
import Linear.Matrix ((!*))
import System.Environment
import System.Exit (exitSuccess)
import System.Random (newStdGen, randomIO, randomRIO, randomRs)

-- linear classification
--

data Neuron = Neuron
  { weights             :: Vector Float
  , bias                :: Float
  , activationFunction  :: Float -> Float
  , activationFunction' :: Float -> Float
  }

type TrainingSet = [(Vector Float, Float)]

mkPerceptron :: Int -> IO Neuron
mkPerceptron numInputs =
  do gen <- newStdGen
     let weights' = take numInputs $ randomRs (-0.1, 0.1) gen
         bias = 0.1
     return $ Neuron (Vector.fromList weights') bias heavisideStep (error "heavisideStep does not have a derivative")  -- id -- sigmoid -- heavisideStep --


mkLinearPerceptron :: Int -> IO Neuron
mkLinearPerceptron numInputs =
  do gen <- newStdGen
     let weights' = take numInputs $ randomRs (-0.4, 0.4) gen
         bias = 0.01
     return $ Neuron (Vector.fromList weights') bias id (const 1) -- sigmoid -- heavisideStep --

mkSigmoidPerceptron :: Int -> IO Neuron
mkSigmoidPerceptron numInputs =
  do gen <- newStdGen
     let weights' = take numInputs $ randomRs (-0.1, 0.1) gen
         bias = 0.01
     return $ Neuron (Vector.fromList weights') bias sigmoid sigmoid' -- sigmoid -- heavisideStep --

heavisideStep :: Float -> Float
heavisideStep x
  | x <= 0     = 0
  | otherwise = 1

sigmoid :: Float -> Float
sigmoid z =
  (1.0 / (1.0 + exp (-z)))

sigmoid' :: Float -> Float
sigmoid' z = (sigmoid z) * (1 - sigmoid z)

z :: Neuron -> Vector Float -> Float
z (Neuron weights bias _ _) inputs = ((dot inputs weights) + bias)

a :: Neuron -> Vector Float -> Float
a = evalNeuron

evalNeuron :: Neuron -> Vector Float -> Float
evalNeuron n@(Neuron weights bias activationFunction _) inputs =
  activationFunction (z n inputs)
--  activationFunction ((dot inputs weights) + bias)

train :: (Vector Float, Float) -> Neuron -> Neuron
train (input, target) neuron =
  let lr = 0.01 -- 0.005
      guess = evalNeuron neuron input
      err   = target - guess
      weights' = (weights neuron) ^+^ (input ^* (err * lr))
      bias' = (bias neuron) + (err * lr)
  in neuron { weights = weights', bias = bias' }

-- The 1/2 just makes it easier to calculate the derivative of the cost later
-- since we multiple by a learning rate anyway, might as well have the 1/2 here to simplify things
cost :: Float -> Float -> Float
cost target current = (1/2) * (target - current) ^ 2


cost0 :: Neuron -> (Vector Float, Float) -> Float
cost0 neuron (inputs, y) =
  let a' = a neuron inputs
  in cost y a'



𝛿C0_𝛿a neuron (inputs, y) = 2 * (a neuron inputs - y)

𝛿C_𝛿a neuron trainingData =
  (sum (map (𝛿C0_𝛿a neuron) trainingData)) / (fromIntegral$ length trainingData)

𝛿a_𝛿z neuron (inputs, y) = sigmoid' (z neuron inputs)

-- 𝛿z_𝛿w neuron (inputs, y) = a neuron inputs
𝛿z_𝛿w neuron (inputs, y) = inputs

𝛿z_𝛿b neuron (inputs, y) = 1

𝛿z_𝛿wk neuron (inputs, y) k = (inputs!k)

𝛿C0_𝛿wk :: Neuron -> Int -> (Vector Float, Float) -> Float
𝛿C0_𝛿wk neuron k x@(input, y) = (𝛿z_𝛿wk neuron x k) * (𝛿a_𝛿z neuron x) * (𝛿C0_𝛿a neuron x)

𝛿C_𝛿wk :: Neuron -> [(Vector Float, Float)] -> Int -> Float
𝛿C_𝛿wk neuron trainingData k =
  sum (map (𝛿C0_𝛿wk neuron k) trainingData) / (fromIntegral (length trainingData))

𝛿C0_𝛿b :: Neuron -> (Vector Float, Float) -> Float
𝛿C0_𝛿b neuron x@(input, y) = (𝛿z_𝛿b neuron x) * (𝛿a_𝛿z neuron x) * (𝛿C0_𝛿a neuron x)

𝛿C_𝛿b :: Neuron -> [(Vector Float, Float)] -> Float
𝛿C_𝛿b neuron trainingData =
  sum (map (𝛿C0_𝛿b neuron) trainingData) / (fromIntegral (length trainingData))

--  (𝛿z_𝛿wk neuron x k) * (𝛿a_𝛿z neuron x) * (𝛿C0_𝛿a neuron x)
{-
𝛿C0_𝛿w neuron x = (𝛿z_𝛿w neuron x) * (𝛿a_𝛿z neuron x) * (𝛿C0_𝛿a neuron x)

𝛿C0_𝛿w neuron trainingData =
  let n = length trainingData
  in sum (map (𝛿C0_𝛿w neuron) trainingData) / n
-}

-- * Delta Rulen
-- https://en.wikipedia.org/wiki/Delta_rule


-- error for a single neuron and a single training data point
e :: Neuron -> (Vector Float, Float) -> Float
e neuron (inputs, target) = 0.5 * (target - (evalNeuron neuron inputs)) ^ 2

h :: Neuron -> Vector Float -> Float
h (Neuron weights bias _ _) inputs = ((dot inputs weights) + bias)

-- change in error with respect to change in a specific weight
-- dE_dwi :: Neuron -> (Vector Float, Float) -> Float

d_w_i :: Neuron -> (Vector Float, Float) -> Int -> Float
d_w_i neuron (inputs, target) i =
  let err = (target - (evalNeuron neuron inputs))
      sp  = (activationFunction' neuron) (h neuron inputs)
      dwi = err * sp * (inputs ! i)
  in {- trace ("i = " ++ show i ++" err = " ++ show err ++ " sp = " ++ show sp ++ " input = " ++ (show $ inputs ! i) ++ " dwi = " ++ show dwi) -} dwi

d_b :: Neuron -> (Vector Float, Float) -> Float
d_b neuron (inputs, target) =
  let err = (target - (evalNeuron neuron inputs))
      sp = (activationFunction' neuron) (h neuron inputs)
      db =  err * sp
  in {- trace ("err = " ++ show err ++ " sp = " ++ show sp ++ " db = " ++ show db) -} db


d_w_v :: Neuron -> (Vector Float, Float) -> Vector Float
d_w_v neuron (inputs, target) =
  let err = (target - (evalNeuron neuron inputs))
      sp  = (activationFunction' neuron) (h neuron inputs)
      dw  = err * sp *^ inputs
  in dw

trainDeltaV :: (Vector Float, Float) -> Neuron -> Neuron
trainDeltaV (inputs, target) neuron =
  let weights' = updateWeights (weights neuron)
      bias'    = updateBias (bias neuron)
  in neuron { weights = weights', bias = bias' }
  where
    alpha = 0.4
    updateWeights w = w ^+^ alpha *^ (d_w_v neuron (inputs, target))
    updateBias b   = b + alpha   * (d_b   neuron (inputs, target))

trainDelta :: (Vector Float, Float) -> Neuron -> Neuron
trainDelta (inputs, target) neuron =
  let weights' = Vector.imap updateWeight (weights neuron)
      bias'    = updateBias (bias neuron)
  in neuron { weights = weights', bias = bias' }
  where
    alpha = 0.4
    updateWeight i w = w + alpha * (d_w_i neuron (inputs, target) i)
    updateBias b     = b + alpha   * (d_b   neuron (inputs, target))

trainDeltaBatch' :: Neuron -> (Vector Float, Float) -> (Vector Float, Float)
trainDeltaBatch' neuron (inputs, target) =
  let {- lr = 0.01 -- 0.005
      guess = evalNeuron neuron input
      err   = target - guess
      weights' = (weights neuron) ^+^ (input ^* (err * lr))
      bias' = (bias neuron) + (err * lr)
      -}
      weights' = Vector.imap updateWeight (weights neuron)
      bias' = updateBias (bias neuron)
  in (weights', bias')
  where
--    alpha = 0.01
    updateWeight i w = (d_w_i neuron (inputs, target) i)
    updateBias b     = (d_b neuron (inputs, target))

trainDeltaBatchV' :: Neuron -> (Vector Float, Float) -> (Vector Float, Float)
trainDeltaBatchV' neuron (inputs, target) =
  let {- lr = 0.01 -- 0.005
      guess = evalNeuron neuron input
      err   = target - guess
      weights' = (weights neuron) ^+^ (input ^* (err * lr))
      bias' = (bias neuron) + (err * lr)
      -}
      weights' = updateWeights (weights neuron)
      bias' = updateBias (bias neuron)
  in (weights', bias')
  where
--    alpha = 0.01
    updateWeights w = (d_w_v neuron (inputs, target))
    updateBias b   = (d_b neuron (inputs, target))

trainDeltaBatch :: [(Vector Float, Float)] -> Neuron -> Neuron
trainDeltaBatch trainingData neuron =
  let (weights'', biases) = unzip (map (trainDeltaBatch' neuron) trainingData)
      alpha = 0.3
      weight_avg = ((sumV weights''))  ^/ (fromIntegral $ length weights'')
      weights'   = (weights neuron) ^+^ (alpha *^ ((sumV weights'') ^/ (fromIntegral $ length weights'')))-- (weights neuron) ^+^ (0.1 *^ (sumV weights))
      bias'      = (bias neuron) + (alpha * ((sum biases) / (fromIntegral $ length  biases)))
  in trace ("weight_avg = " ++ show weight_avg) $
       neuron { weights = weights'
              , bias = bias'
              }


quadratic_cost_1 :: Neuron -> (Vector Float, Float) -> Float
quadratic_cost_1 neuron (inputs, target) =
  let guess = evalNeuron neuron inputs
  in cost target guess -- (target - guess) ^ 2

quadratic_cost :: Neuron -> [(Vector Float, Float)] -> Float
quadratic_cost neuron training =
  let len = fromIntegral (length training)
  in (sum (map (cost0 neuron) training)) / (2 * len)

trainCost :: Neuron -> [(Vector Float, Float)] -> Neuron
trainCost neuron trainingData =
  let weights' = Vector.imap trainWeight (weights neuron)
      bias' = trainBias (bias neuron)
  in neuron { weights = weights'
            , bias    = bias'
            }
  where
    trainingRate = 0.4

    trainWeight :: Int -> Float -> Float
    trainWeight k w = w - trainingRate * (𝛿C_𝛿wk neuron trainingData k)

    trainBias :: Float -> Float
    trainBias b = b - trainingRate * (𝛿C_𝛿b neuron trainingData)


-- cost_derivative :: Float -> Float -> Float
-- cost_derivative

p0 = Neuron (Vector.fromList [6, 2, 2]) (-3) heavisideStep undefined
p1 = Neuron (Vector.fromList [6, 2, 2]) 5 heavisideStep undefined

inputs0 :: Vector Float
inputs0 = Vector.fromList [ 0, 1, 1]

windowRadius = 800

window :: Display
window = InWindow "bad idea" (windowRadius,windowRadius) (10,10)

-- convert from (-1,1) to window coord
toCoord :: Float -> Float
toCoord n = n * ( ((fromIntegral windowRadius) - 100) / 2)

background :: Color
background = white

markerRadius = 5

drawX :: Picture
drawX =
--   color c $
--   translate x y $
    pictures [ line [(-markerRadius, -markerRadius), (markerRadius, markerRadius)]
             , line [(-markerRadius, markerRadius), (markerRadius, -markerRadius)]
             ]

drawO :: Picture
drawO =
--  color c $
--   translate x y $
    circle markerRadius

render :: World -> Picture
render (World neuron trainingData index _) =
  pictures (box : center : xLabel : yLabel : renderNeuron neuron : [ renderTd td i | (td, i) <- zip trainingData [0..] ])
  where
    center :: Picture
    center = color black $ pictures $ [ line [ (-5, 0), (5, 0) ], line [ (0, -5), (0, 5) ] ]
    xLabel = translate (toCoord (-0.1)) (toCoord (-1.1)) $ scale 0.1 0.1 $ Text "Hotness"
    yLabel = translate (toCoord (-1.1)) (toCoord (-0.1)) $ rotate (-90) $ scale 0.1 0.1 $ Text "$$$"
    box =  line [ (toCoord (-1), toCoord (-1))
                , (toCoord (1), toCoord (-1))
                , (toCoord (1), toCoord 1)
                , (toCoord (-1), toCoord 1)
                , (toCoord (-1), toCoord (-1))
                ]
    renderNeuron :: Neuron -> Picture
    renderNeuron n =
      let m = (weights n ! 0) / (weights n ! 1)
          x = 1
          y x' = (- ((bias n) / (weights n ! 1))) - m * x'
      in line [ (toCoord (-x), toCoord $ y (-x)), (toCoord x, toCoord $ y x) ]
--         line [ (toCoord (-1), toCoord (-1)), (toCoord 1, toCoord 1) ]
    renderTd :: (Vector Float, Float) -> Int -> Picture
    renderTd (inputs, expected) i =
      let guessed = evalNeuron neuron inputs
          c = if (i == index) then blue else (if (abs (expected - guessed)) < 0.5 then green else red)
          xc = toCoord (inputs ! 0)
          yc = toCoord (inputs ! 1)
      in color c $
          translate xc yc $
            if expected > 0
               then drawO
               else drawX

fps :: Int
fps = 120

data World = World
 { neuron :: Neuron
 , trainingData :: [(Vector Float, Float)]
 , index :: Int
 , isTraining :: Bool
 }

initialState :: IO World
initialState =
  do n <- mkPerceptron 2
     g  <- newStdGen
     g' <- newStdGen
     let xs = randomRs (-1, 1::Float) g
         ys = randomRs (-1, 1::Float) g'
         td = take 50 $ zipWith (\x y -> (Vector.fromList [x, y], if (y > -(0.5*x) + 0.3) then 1 else 0)) xs ys
     pure $ World n td 0 False

initialStateLinear :: IO World
initialStateLinear =
  do n <- mkLinearPerceptron 2
     g  <- newStdGen
     g' <- newStdGen
     let xs = randomRs (-1, 1::Float) g
         ys = randomRs (-1, 1::Float) g'
         td = take 50 $ zipWith (\x y -> (Vector.fromList [x, y], if (y > -(0.5*x) + 0.3) then 1 else 0)) xs ys
     pure $ World n td 0 False

initialStateSigmoid :: IO World
initialStateSigmoid =
  do n <- mkSigmoidPerceptron 2
     g  <- newStdGen
     g' <- newStdGen
     let xs = randomRs (-1, 1::Float) g
         ys = randomRs (-1, 1::Float) g'
         td = take 150 $ zipWith (\x y -> (Vector.fromList [x, y], if (y > -(0.5*x) + 0.3) then 1 else 0)) xs ys
--         td = take 50 $ zipWith (\x y -> (Vector.fromList [x, y], sigmoid (y - (-(0.5*x) + 0.3)))) xs ys
     pure $ World n td 0 False

andState :: IO World
andState =
    do n <- mkPerceptron 2
       let trainingState = [ (Vector.fromList [1,1], 1)
                           , (Vector.fromList [1, (-1)], (-1))
                           , (Vector.fromList [(-1), (-1)], (-1))
                           , (Vector.fromList [(-1), (1)], (-1))
                           ]
       pure $ World n trainingState 0 False

orState :: IO World
orState =
    do n <- mkPerceptron 2
       let trainingState = [ (Vector.fromList [1   ,1    ],   1)
                           , (Vector.fromList [1   , (-1)],  (1))
                           , (Vector.fromList [(-1), (-1)], (-1))
                           , (Vector.fromList [(-1), (1) ], (1))
                           ]
       pure $ World n trainingState 0 False

xorState :: IO World
xorState =
    do n <- mkPerceptron 2
       let trainingState = [ (Vector.fromList [1   ,1    ],   (-1))
                           , (Vector.fromList [1   , (-1)],  (1))
                           , (Vector.fromList [(-1), (-1)], (-1))
                           , (Vector.fromList [(-1), (1) ], (1))
                           ]
       pure $ World n trainingState 0 False

handleInput :: Event -> World -> World
handleInput event world =
  case event of
    (EventKey _ Up _ _) ->
      world { isTraining = not (isTraining world) }
    _ -> world

isTrainingComplete :: Neuron -> TrainingSet -> Bool
isTrainingComplete n td =
  all (\(inputs, expected) -> evalNeuron n inputs == expected) td

update :: Float -> World -> World
update delta w =
  if isTraining w then
    let n' = train ((trainingData w)!!(index w)) (neuron w)
        c = quadratic_cost n' (trainingData w)
    in trace (show c) $ w { neuron = n'
         , index = ((index w) + 1) `mod` (length (trainingData w))
         , isTraining = not (isTrainingComplete n' (trainingData w))
         }
  else w


-- online vs batch updates?
updateDelta :: Float -> World -> World
updateDelta delta w =
  if isTraining w then
    let n' = trainDeltaV ((trainingData w)!!(index w)) (neuron w)
        c = quadratic_cost n' (trainingData w)
    in trace ("weights = " ++ show (weights n') ++ " bias = " ++ show (bias n') ++ " c = " ++ show c) $
       w { neuron = n'
         , index = ((index w) + 1) `mod` (length (trainingData w))
         }
  else w


updateDeltaBatch :: Float -> World -> World
updateDeltaBatch delta w =
  if isTraining w then
    let n' = trainDeltaBatch (trainingData w) (neuron w)
        c = quadratic_cost n' (trainingData w)
    in trace ("weights = " ++ show (weights n') ++ " bias = "++ show (bias n') ++ " c = " ++ show c) $ w { neuron = n'
--                          , index = ((index w) + 1) `mod` (length (trainingData w))
                          }
  else w

updateCost :: Float -> World -> World
updateCost delta w =
  if isTraining w then
    let n' = trainCost (neuron w) (trainingData w)
        c = quadratic_cost n' (trainingData w)
        dc = 𝛿C_𝛿a (neuron w) (trainingData w)
    in trace (show (c, dc)) $
         w { neuron = n'
--           , isTraining = not (isTrainingComplete n' (trainingData w))
           }
  else w


main :: IO ()
main =
  do print $ evalNeuron p0 inputs0
     print $ evalNeuron p1 inputs0
     world <- initialStateSigmoid
     play window background fps world render handleInput updateDeltaBatch

{-
  let n' = foldr train (neuron w) (trainingData w)
  in w { neuron = n' }
-}
{-
p1 = Perceptron (Vector.fromList [3, 1, 1])

-- Vector.fromList[ (Vector.fromList  [6,2,2]), (Vector.fromList  [3,1,1])] !* Vector.fromList [1, 1, 1]

-}
{-
data Layer = Layer
 { perceptrons :: Vector Neuron
 }

type Brain = Vector Layer
-}

{-

deltaRule neuron (inputs, target) i =
  let alpha = 0.1
  in alpha * sigmoid' (z neuron inputs) * inputs!i

-}
