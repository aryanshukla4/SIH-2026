\ A tiny refinery-style blending model, readable by eye.
\ Two crudes blended into one product with a sulphur cap.
Maximize
 profit: 12 light + 9 heavy
Subject To
 supply_light: light <= 100
 supply_heavy: heavy <= 150
 capacity:     light + heavy <= 200
 sulphur:      0.5 light + 2.5 heavy <= 300
 min_run:      light + heavy >= 50
Bounds
 light >= 0
 heavy >= 0
End
