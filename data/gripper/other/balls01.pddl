; Automatically converted to only require STRIPS and negative preconditions

(define (problem strips-gripper-balls-1)
  (:domain gripper-strips)
  (:objects rooma roomb ball1 left right)
  (:init
    (room rooma)
    (room roomb)
    (ball ball1)
    (at-robby rooma)
    (free left)
    (free right)
    (at ball1 rooma)
    (gripper left)
    (gripper right)
  )
  (:goal
    (and
      (at ball1 roomb)
    )
  )
)
