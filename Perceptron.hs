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

type TrainingSet = [(Vector Float, Float)]

mkPerceptron :: Int -> IO Neuron
mkPerceptron numInputs =
  do gen <- newStdGen
     let weights' = take numInputs $ randomRs (-0.1, 0.1) gen
         bias = 0
     return $ Neuron (Vector.fromList weights') bias heavisideStep

heavisideStep :: Float -> Float
heavisideStep x
  | x < 0     = (-1)
  | otherwise = 1

evalNeuron :: Neuron -> Vector Float -> Float
evalNeuron (Neuron weights bias activationFunction) inputs =
  activationFunction ((dot inputs weights) + bias)

train :: (Vector Float, Float) -> Neuron -> Neuron
train (input, target) neuron =
  let lr = 1 -- 0.005
      guess = evalNeuron neuron input
      err   = target - guess
      weights' = (weights neuron) ^+^ (input ^* (err * lr))
      bias' = (bias neuron) + (err * lr)
  in neuron { weights = weights', bias = bias' }

p0 = Neuron (Vector.fromList [6, 2, 2]) (-3) heavisideStep
p1 = Neuron (Vector.fromList [6, 2, 2]) 5 heavisideStep

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
          c = if (i == index) then blue else (if guessed == expected then green else red)
          xc = toCoord (inputs ! 0)
          yc = toCoord (inputs ! 1)
      in color c $
          translate xc yc $
            if expected >= 0
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
     g <- newStdGen
     g' <- newStdGen
     let xs = randomRs (-1, 1::Float) g
         ys = randomRs (-1, 1::Float) g'
         td = take 150 $ zipWith (\x y -> (Vector.fromList [x, y], if (y > -(0.5*x) + 0.3) then 1 else -1)) xs ys
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
    in trace (show $ (bias n', weights n')) $ w { neuron = n'
         , index = ((index w) + 1) `mod` (length (trainingData w))
         , isTraining = not (isTrainingComplete n' (trainingData w))
         }
  else w


main :: IO ()
main =
  do print $ evalNeuron p0 inputs0
     print $ evalNeuron p1 inputs0
     world <- xorState
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
