module Main where

import Data.Vector ((!), Vector)
import qualified Data.Vector as Vector
import Debug.Trace (trace)
import Graphics.Gloss hiding (Vector)
import Graphics.Gloss.Interface.Pure.Game hiding (Vector)
import Linear.Algebra (mult)
import Linear.Metric (dot)
import Linear.Vector (sumV, (^+^),(^*) )
import Linear.Matrix ((!*))
import System.Environment
import System.Exit (exitSuccess)
import System.Random (newStdGen, randomIO, randomRIO, randomRs)

-- linear classification
-- 

data Neuron = Neuron
  { weights            :: Vector Float
  , bias               :: Float
  , activationFunction :: Float -> Float
  }

type TrainingData = [(Vector Float, Float)]

mkPerceptron :: Int -> IO Neuron
mkPerceptron numInputs =
  do gen <- newStdGen
     let weights' = take numInputs $ randomRs (-1, 1) gen
         bias = 1
     return $ Neuron (Vector.fromList weights') bias heavisideStep

heavisideStep :: Float -> Float
heavisideStep x
  | x < 0     = (-1)
  | otherwise = 1

evalNeuron :: Neuron -> Vector Float -> Float
evalNeuron (Neuron weights bias activationFunction) inputs =
  activationFunction ((dot inputs weights) + bias)

train :: (Vector Float, Float) -> Neuron -> Neuron
train (input, expected) neuron =
  let lr = 0.1
      guess = evalNeuron neuron input
      err   = expected - guess
      weights' = (weights neuron) ^+^ (input ^* (err * lr))
      bias' = (bias neuron) + (err * 1000 * lr)
  in neuron { weights = weights', bias = bias' }

p0 = Neuron (Vector.fromList [6, 2, 2]) (-3) heavisideStep
p1 = Neuron (Vector.fromList [6, 2, 2]) 5 heavisideStep

inputs0 :: Vector Float
inputs0 = Vector.fromList [ 0, 1, 1]

windowRadius = 1000

window :: Display
window = InWindow "bad idea" (windowRadius,windowRadius) (10,10)

background :: Color
background = white

markerRadius = 5

drawX :: Color -> (Float, Float) -> Picture
drawX c (x,y) =
  color c $
   translate x y $
    pictures [ line [(-markerRadius, -markerRadius), (markerRadius, markerRadius)]
             , line [(-markerRadius, markerRadius), (markerRadius, -markerRadius)]
             ]

drawO :: Color -> (Float, Float) -> Picture
drawO c (x,y) =
  color c $
   translate x y $
    circle markerRadius

render :: World -> Picture
render (World neuron trainingData index _) =
  pictures (center : renderNeuron neuron : [ renderTd td i | (td, i) <- zip trainingData [0..] ])
  where
    center :: Picture
    center = color black $ circle 5
    renderNeuron :: Neuron -> Picture
    renderNeuron n =
      let m = (weights n ! 0) / (weights n ! 1)
          y x' = - ((bias n) / (weights n ! 1)) - m * x'
          x = fromIntegral windowRadius
      in line [ (-x, y (-x)), (x, y x) ]
    renderTd :: (Vector Float, Float) -> Int -> Picture
    renderTd (inputs, expected) i =
      let guessed = evalNeuron neuron inputs
          color = if (i == index) then blue else (if guessed == expected then green else red)
      in if expected >= 0 then drawO color ((inputs ! 0), (inputs ! 1)) else drawX color ((inputs ! 0), (inputs ! 1))

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
     g <- newStdGen
     g' <- newStdGen
     let xs = randomRs (-400, 400::Int) g
         ys = randomRs (-400, 400::Int) g'
         td = take 150 $ zipWith (\x y -> (Vector.fromList [fromIntegral x, fromIntegral y], if (y > -(2*x) + 100) then 1 else -1)) xs ys
     pure $ World n td 0 False

handleInput :: Event -> World -> World
handleInput event world =
  case event of
    (EventKey _ Up _ _) ->
      world { isTraining = not (isTraining world) }
    _ -> world

isTrainingComplete :: Neuron -> TrainingData -> Bool
isTrainingComplete n td =
  all (\(inputs, expected) -> evalNeuron n inputs == expected) td

update :: Float -> World -> World
update delta w =
  if isTraining w then
    let n' = train ((trainingData w)!!(index w)) (neuron w)
    in trace (show $ (bias n', weights n')) $ w { neuron = n'
         , index = ((index w) + 1) `mod` (length (trainingData w))
         , isTraining = not (isTrainingComplete n' (trainingData w))
         }
  else w


main :: IO ()
main =
  do print $ evalNeuron p0 inputs0
     print $ evalNeuron p1 inputs0
     world <- initialState
     play window background fps world render handleInput update

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
 
