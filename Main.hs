module Main where

import Control.Monad (replicateM)
-- import Control.Concurrent.STM
import Data.Attoparsec.ByteString(IResult(..), parse)
import Data.ByteString (hGetSome)
import Data.MNIST
import Data.Vector (Vector)
import qualified Data.Vector as Vector
import System.Environment
import System.IO (openFile, IOMode(ReadMode))
import System.Random (randomIO)

main :: IO ()
main =
  do [fp] <- getArgs
     train fp

train :: FilePath -> IO ()
train fp =
  do h <- openFile fp ReadMode
     bs <- hGetSome h 4
     case parse magicWord bs of
      (Done _ (size, dims)) ->
        do bs <- hGetSome h (4 * (fromIntegral dims))
           case parse (sizes dims) bs of
            (Done _ szs) ->
              do print (size, dims, szs)

data Network = Network
  { biases :: Vector Double
  , weights :: Vector (Vector (Vector Double))
  }
  deriving (Eq, Ord, Read, Show)


feedforward :: Network -> Double -> Vector Double
feedforward network a =
 undefined

initNetwork :: [Int] -> IO Network
initNetwork szs =
  do bs <- Vector.fromList <$> (mapM (const randomIO) (tail szs))
     ws <- Vector.fromList <$>
       mapM (\(j,k) -> Vector.fromList <$> replicateM j (Vector.fromList <$> replicateM k randomIO)) (zip (tail szs) szs)
     pure $ Network { biases = bs
                    , weights = ws
                    }

{-
  do bs' <- mapM (const randomIO) (tail let)
     sizes bs = Vector.fromList (0:bs')
     ws <- Vector.fromList <$> mapM (\sz -> Vector.fromList <$> replicateM sz randomIO) sizes
     pure $ Network { biases = bs
                    , weights = ws
                    }
-}
--  Network { biases = 

sigmoid z =
  1.0 / (1.0 + exp (-z))


