var DEBUG = false;
var ISMAP = false;
function Cell(i, j, w) {
  this.i = i;
  this.j = j;
  this.x = i * w;
  this.y = j * w;
  this.w = w;

  this.revealed = false;
  this.over = false;

  this.block = false;
  this.agent = false;
  this.other_agent = false;
  this.goal = '';
  this.value = 0;
  this.id = 0;
}

Cell.prototype.show = function () {
  stroke(192,192,192);
  noFill();
  rect(this.x, this.y, this.w, this.w);
  if (this.revealed) {
    fill(127);
    // Add blocks
    if (this.block) {
      fill(0);
      rect(this.x, this.y, this.w, this.w);
    }

    if (this.goal == 'wall') {
      fill(128, 128, 128);
      rect(this.x, this.y, this.w, this.w);
    }
    else if (this.goal == 'door') {
      fill(128, 0, 128); //purple
      rect(this.x, this.y, this.w, this.w);
    }
    else if (this.goal == 'yellow') {
      fill(255,215,0);//gold
      rect(this.x, this.y, this.w, this.w);
    } 
    else if (this.goal == 'green') {
      fill(0, 128, 0);
      rect(this.x, this.y, this.w, this.w);
    }
    else if (this.goal == 'blue') {
      fill(0, 0, 255);
      rect(this.x, this.y, this.w, this.w);
    }
    else if (this.goal == 'red') {
      fill(255, 0, 0);
      rect(this.x, this.y, this.w, this.w);
    }
    else if (this.goal == 'rubble') {
      fill(182,182,182);
      rect(this.x, this.y, this.w, this.w);
      textAlign(CENTER);
      fill(0);
      text("X", this.x + this.w * 0.15, this.y + this.w * 0.9, this.w); 
    }

    // comment out to show the other agents even they're out of FOV
    // else if (this.other_agent) {
    //   this.addOtherAgent(this.id);
    // }
  }

  if (this.over) {
    fill(127);
    rect(this.x, this.y, this.w, this.w);
  }
  
  // Add borders
  if (this.goal == 'borders') {
    // fill(200,200,200,127);
    fill(173,216,230,127);
    rect(this.x, this.y, this.w, this.w);
  }
  if (DEBUG) {
    // Add blocks
    if (this.block) {
      fill(0);
      rect(this.x, this.y, this.w, this.w);
    }
    if (this.goal == 'wall') {
      fill(128, 128, 128);
      rect(this.x, this.y, this.w, this.w);
    }
    else if (this.goal == 'door') {
      fill(128, 0, 128);
      rect(this.x, this.y, this.w, this.w);
    }
    else if (this.goal == 'yellow') {
      fill(255,215,0);
      rect(this.x, this.y, this.w, this.w);
    } 
    else if (this.goal == 'green') {
      fill(0, 128, 0);
      rect(this.x, this.y, this.w, this.w);
    }
    else if (this.goal == 'blue') {
      fill(0, 0, 255);
      rect(this.x, this.y, this.w, this.w);
    }
    else if (this.goal == 'red') {
      fill(255, 0, 0);
      rect(this.x, this.y, this.w, this.w);
    }
    else if (this.goal == 'rubble') {
      fill(182,182,182);
      rect(this.x, this.y, this.w, this.w);
      textAlign(CENTER);
      fill(0);
      text("X", this.x + this.w * 0.15, this.y + this.w * 0.9, this.w); 
    }
    // else if (this.other_agent) {
    //   this.addOtherAgent(this.id);
    // }
  }
  if (ISMAP) {
    if (this.block) {
      fill(0);
      rect(this.x, this.y, this.w, this.w);
    }
    if (this.goal == 'wall') {
      fill(128, 128, 128);
      rect(this.x, this.y, this.w, this.w);
    }
    else if (this.goal == 'door') {
      fill(128, 0, 128);
      rect(this.x, this.y, this.w, this.w);
    }
  }
}

Cell.prototype.addAgent = function () {
  this.agent = true;
  fill(255,83,73);
  ellipse(this.x + this.w * 0.5, this.y + this.w * 0.5, this.w);
}

Cell.prototype.addMyAgent = function (agent_id) {
  this.agent = true;
  // fill(255,83,73);
  fill(128,0,0);
  ellipse(this.x + this.w * 0.5, this.y + this.w * 0.5, this.w);
  textAlign(CENTER);
  fill(255);
  text(agent_id, this.x + this.w * 0.1, this.y + this.w * 0.8, this.w); 
}

Cell.prototype.addAgentImage = function (roleName) {
  this.agent = true;
  if (roleName=='medic'){
    image(medicImg, this.x, this.y, 15, 15)
  }else if (roleName=='engineer'){
    image(engineerImg, this.x, this.y, 15, 15)

  }
  
}

Cell.prototype.addOtherAgent = function (agent_id) {
  // this.other_agent = true;
  // fill(123,0,0); //change back
  // fill(255,83,73);
  fill(0,0,128);
  ellipse(this.x + this.w * 0.5, this.y + this.w * 0.5,this.w);
  textAlign(CENTER);
  fill(255);
  text(agent_id, this.x + this.w * 0.1, this.y + this.w * 0.8, this.w); 
}

Cell.prototype.addOtherAgentImg = function (roleName) {
  // fill(255,63,47);
  // fill(105,105,105);
  if (roleName=='medic'){
    image(medicImg, this.x, this.y, 15, 15)
  }else if (roleName=='engineer'){
    image(engineerImg, this.x, this.y, 15, 15)
  }
}

Cell.prototype.markAsVisitedbyAgent = function () {
  this.agent = false;
  fill(0,0,255); //change back
  // fill(255,83,73);
  ellipse(this.x + this.w * 0.5, this.y + this.w * 0.5,this.w/3);
  // textAlign(CENTER);
  // fill(255);
  // text(val, this.x + this.w * 0.1, this.y + this.w * 0.8, this.w); 
}

Cell.prototype.markAsVisitedbyRole = function (roleName) {
  this.agent = false;
  if (roleName=='medic'){
    fill(229,83,0); 
    ellipse(this.x + this.w * 0.5, this.y + this.w * 0.5, this.w/2);
  }else if (roleName=='engineer'){
    fill(0,0,255); //change back
    ellipse(this.x + this.w * 0.5, this.y + this.w * 0.5, this.w/2);
  }
    
  // fill(255,83,73);
  
  // textAlign(CENTER);
  // fill(255);
  // text(val, this.x + this.w * 0.1, this.y + this.w * 0.8, this.w); 
}

Cell.prototype.markAsVisitedbyEngineer = function () {
  this.agent = false;
  fill(255,215,0); 
  ellipse(this.x + this.w * 0.5, this.y + this.w * 0.5, this.w/3);
}



Cell.prototype.drawFoV = function () {
  fill(250,250,250,63);
  rect(this.x, this.y, this.w, this.w);
}

Cell.prototype.addText = function (val){
  textAlign(CENTER);
  fill(0);
  text(val, this.x + this.w * 0.5, this.y + this.w - 15);
}

Cell.prototype.contains = function (x, y) {
  return (x > this.x && x < this.x + this.w && y > this.y && y < this.y + this.w);
}

Cell.prototype.reveal = function () {
  this.revealed = true;
}