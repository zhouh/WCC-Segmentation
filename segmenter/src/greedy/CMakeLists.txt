INCLUDE_DIRECTORIES(${SOURCE_DIR})
ADD_EXECUTABLE(greedy main.cc greedy.cc ../corpus.cc ../logging.cc ../training_utils.cc ../utils.cc )
target_link_libraries(greedy cnn glog ${Boost_LIBRARIES})
