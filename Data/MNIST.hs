{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}
module Data.MNIST where

import Control.Monad
import Control.DeepSeq
-- import Data.Attoparsec
import Data.Attoparsec.Binary
import Data.Attoparsec.ByteString
import Data.Int (Int32)
import Data.Word
import Data.Vector (Vector, fromList)
import qualified Data.Vector.Unboxed as U
import GHC.TypeLits

data DataType
  = UnsignedByte
  | SignedByte
  | Short
  | Int
  | Float
  | Double
    deriving (Eq, Ord, Read, Show)

data MagicWord = MagicWord
  { dataType :: DataType
  , dims  :: Word8 -- ^ number of dimesions. e.g. 1 = vector, 2 = 2D matrix, 3 = 3D matrix
  }
  deriving (Eq, Ord, Read, Show)

data family MNIST (dataType :: DataType) (dim :: Nat) :: *
data instance MNIST UnsignedByte 1 = UnsignedByteV1  Word32  !(U.Vector Word8) deriving Show
-- data instance MNIST UnsignedByte 2 = UnsignedByteV2 (Word32, Word32) (Vector (U.Vector Word8)) deriving Show
data instance MNIST UnsignedByte 3 = UnsignedByteV3 !(Word32, Word32, Word32) !(Vector (Vector (U.Vector Word8))) deriving Show

pDataType :: Parser DataType
pDataType =
  do s <- anyWord8
     case s of
       0x08 -> pure UnsignedByte
       0x09 -> pure SignedByte
       0x0B -> pure Short
       0x0C -> pure Int
       0x0D -> pure Float
       0x0E -> pure Double
       _    -> fail $ show s ++ "is not a valid DataType in the magic word"

-- | (data size, number of dimensions)
pMagicWord :: Parser MagicWord
pMagicWord =
  do word8 0
     word8 0
     s <- pDataType
     dim <- anyWord8
     pure $ MagicWord s dim

pSizes :: Word8 -> Parser [Word32]
pSizes dims = replicateM (fromIntegral dims) anyWord32be

pMNISTUnsignedByteV1 :: Parser (MNIST UnsignedByte 1)
pMNISTUnsignedByteV1 =
  do (MagicWord dt dims) <- pMagicWord
     if (dt == UnsignedByte) && (dims == 1)
       then do [size] <- pSizes dims
               ds <- pUnsignedByteV1 size
               pure $ UnsignedByteV1 size ds
       else error "unexpected data type or dimension"
{-
pMNISTUnsignedByteV2 :: Parser (MNIST UnsignedByte 2)
pMNISTUnsignedByteV2 =
  do (MagicWord dt dims) <- pMagicWord
     if (dt == UnsignedByte) && (dims == 2)
       then do [size0, size1] <- pSizes dims
               ds <- pUnsignedByteV2 size0 size1
               pure $ UnsignedByteV2 (size0, size1) ds
       else error "unexpected data type or dimension"
-}
pMNISTUnsignedByteV3 :: Parser (MNIST UnsignedByte 3)
pMNISTUnsignedByteV3 =
  do (MagicWord dt dims) <- pMagicWord
     if (dt == UnsignedByte) && (dims == 3)
       then do [size0, size1, size2] <- pSizes dims
               ds <- pUnsignedByteV3 size0 size1 size2 <* endOfInput
               pure $ UnsignedByteV3 (size0, size1, size2) (force ds)
       else error "unexpected data type or dimension"

pUnsignedByteV1 :: Word32 -> Parser (U.Vector Word8)
pUnsignedByteV1 c =
  (fmap (force . U.fromList)) $! replicateM (fromIntegral c) anyWord8

pUnsignedByteV2 :: Word32 -> Word32 -> Parser (Vector (U.Vector Word8))
pUnsignedByteV2 x y =
  (fmap (force . fromList)) $! replicateM (fromIntegral x) (pUnsignedByteV1 y)

pUnsignedByteV3 :: Word32 -> Word32 -> Word32 -> Parser (Vector (Vector (U.Vector Word8)))
pUnsignedByteV3 x y z =
  (fmap (force . fromList)) $! replicateM (fromIntegral x) (pUnsignedByteV2 y z)

