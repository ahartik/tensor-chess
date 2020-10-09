#ifndef _CHESS_MODEL_COLLECTION_H_
#define _CHESS_MODEL_COLLECTION_H_

#include "generic/model.h"

namespace chess {

extern const char kModelPath[];
generic::ModelCollection* GetModelCollection();

}  // namespace chess

#endif
