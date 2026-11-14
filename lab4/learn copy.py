import json
import random
import numpy as np
import concurrent.futures
import multiprocessing
from tqdm.auto import tqdm

def init_states():
    settings = dict(
        width = 375,
        height = 667,
        platformWidth = 65,
        platformHeight = 20,

        gravity = 0.33,
        drag = 0.3,
        bounceVelocity = -12.5,

        minPlatformSpace = 15,
        maxPlatformSpace = 20,
        keydown = False,
        score = 0,
    )
    settings['platformStart'] = settings['height'] - 50

    platforms = [dict(x=settings['width'] / 2 - settings['platformWidth'] / 2,
                      y=settings['platformStart'])]
    y = settings['platformStart']
    while (y > 0):
        # ИСПРАВЛЕНО: Строгое вычитание с uniform (без randint)
        y -= (settings['platformHeight'] + 
              np.random.uniform(settings['minPlatformSpace'], 
                                settings['maxPlatformSpace']))

        while True:
            x = np.random.uniform(25, settings['width'] - 25 - settings['platformWidth'])
            if not ((y > settings['height'] / 2) and
                    (x > settings['width'] / 2 - settings['platformWidth'] * 1.5) and
                    (x < settings['width'] / 2 + settings['platformWidth'] / 2)):
                break
        platforms.append(dict(x=x, y=y))

    doodle = dict(
      width=40,
      height=60,
      x=settings['width'] / 2 - 20,
      y=settings['platformStart'] - 60,
      dx=0,
      dy=0,
      playerDir=0,
      prevDoodleY=settings['platformStart'] - 60,
    )

    return doodle, platforms, settings

def restart():
    return init_states()

def loop(doodle, platforms, settings):
    doodle['dy'] += settings['gravity']

    if (doodle['y'] < settings['height'] / 2 and doodle['dy'] < 0):
        for i, _ in enumerate(platforms):
            platforms[i]['y'] -= doodle['dy']

        while (platforms[-1]['y'] > 0):
            # ИСПРАВЛЕНО: Убран багованный внешний np.random.uniform
            platforms.append(dict(
                x=np.random.uniform(25,
                                    settings['width'] - 25 - settings['platformWidth']),
                y=platforms[-1]['y'] - (
                    settings['platformHeight'] +
                    np.random.uniform(
                        settings['minPlatformSpace'],
                        settings['maxPlatformSpace']
                    )
                )
            ))

            settings['minPlatformSpace'] = min(settings['minPlatformSpace'] + 0.5,
                                               settings['height'] / 2 - 0.5)
            settings['maxPlatformSpace'] = min(settings['maxPlatformSpace'] + 0.5,
                                               settings['height'] / 2)
    else:
        doodle['y'] += doodle['dy']

    if not settings['keydown']:
        if (doodle['playerDir'] < 0):
            doodle['dx'] += settings['drag']
            if (doodle['dx'] > 0):
                doodle['dx'] = 0
                doodle['playerDir'] = 0
        elif (doodle['playerDir'] > 0):
            doodle['dx'] -= settings['drag']

            if (doodle['dx'] < 0):
                doodle['dx'] = 0
                doodle['playerDir'] = 0

    doodle['x'] += doodle['dx']

    if (doodle['x'] + doodle['width'] < 0):
        doodle['x'] = settings['width']
    elif (doodle['x'] > settings['width']):
        doodle['x'] = -doodle['width']

    for platform in platforms:
        if (
          (doodle['dy'] > 0) and
          (doodle['prevDoodleY'] + doodle['height'] <= platform['y']) and
          (doodle['x'] < platform['x'] + settings['platformWidth']) and
          (doodle['x'] + doodle['width'] > platform['x']) and
          (doodle['y'] < platform['y'] + settings['platformHeight']) and
          (doodle['y'] + doodle['height'] > platform['y'])
        ):
            doodle['y'] = platform['y'] - doodle['height']
            doodle['dy'] = settings['bounceVelocity']

    doodle['prevDoodleY'] = doodle['y']
    platforms_cleared = len(platforms)
    platforms[:] = [p for p in platforms if p['y'] < settings['height']]
    platforms_cleared -= len(platforms)
    settings['score'] += platforms_cleared

    if doodle['y'] > settings['height'] + doodle['height']:
        return -1
    return 0

def apply_action(doodle, platforms, settings, action):
    action = np.clip(action, -1.0, 1.0)

    # Мертвая зона
    if abs(action) < 0.1:
        settings['keydown'] = False
        return

    settings['keydown'] = True
    doodle['playerDir'] = -1 if action < 0 else 1
    doodle['dx'] = action * 5.0
    
def get_features(doodle, platforms, settings):
    screen_width = settings['width']
    screen_height = settings['height']
    
    d_center_x = doodle['x'] + doodle['width'] / 2
    d_feet_y = doodle['y'] + doodle['height']
    
    # 4 базовые фичи
    feats = [
        d_center_x / screen_width,
        d_feet_y / screen_height,
        doodle['dx'] / 5.0,  # Нормализованная скорость X
        doodle['dy'] / 15.0  # Нормализованная скорость Y (важно: показывает, падаем или летим вверх)
    ]
    
    below = []
    above = []
    
    for p in platforms:
        px = p['x'] + settings['platformWidth'] / 2
        py = p['y']
        
        # Горизонтальное расстояние С УЧЕТОМ ПЕРЕХОДА ЭКРАНА
        dx = px - d_center_x
        if dx > screen_width / 2: dx -= screen_width
        elif dx < -screen_width / 2: dx += screen_width
        
        dy = py - d_feet_y
        
        # ЭВРИСТИКА ДОСТИЖИМОСТИ: 
        # Штрафуем платформы, которые находятся далеко по горизонтали.
        # Умножаем dx на 1.5, так как лететь вбок сложнее/дольше, чем падать вниз.
        reachability_cost = abs(dy) + abs(dx) * 1.5
        
        if dy >= 0:
            below.append((dx / screen_width, dy / screen_height, reachability_cost))
        else:
            above.append((dx / screen_width, dy / screen_height, reachability_cost))
            
    # Сортируем по "стоимости пути" (самые легко достижимые - первые)
    below.sort(key=lambda item: item[2])
    above.sort(key=lambda item: item[2])
    
    # Берем 4 самые доступные платформы СНИЗУ (куда приземляться)
    for i in range(4):
        if i < len(below):
            feats.extend([below[i][0], below[i][1]])
        else:
            feats.extend([0.0, 1.0]) # Заглушка: если платформ нет, эмулируем пол далеко внизу
            
    # Берем 4 самые доступные платформы СВЕРХУ (куда планировать следующий прыжок)
    for i in range(4):
        if i < len(above):
            feats.extend([above[i][0], above[i][1]])
        else:
            feats.extend([0.0, -1.0]) # Заглушка: небо далеко вверху
            
    return np.array(feats)


def get_one(h1=24, h2=12):
    W1 = np.random.normal(size=(20, h1))
    W2 = np.random.normal(size=(h1, h2))
    W3 = np.random.normal(size=(h2, 1))
    return W1, W2, W3

def getAction(doodle, platforms, settings, weights):
    W1, W2, W3 = weights
    x = get_features(doodle, platforms, settings)

    h1 = x.dot(W1)
    h1 = np.where(h1 > 0, h1, 0.05 * h1)
    
    h2 = h1.dot(W2)
    h2 = np.where(h2 > 0, h2, 0.05 * h2)
    
    output = h2.dot(W3)
    return float(np.tanh(output[0]))

def get_score(weights, patience=300):
    doodle, platforms, settings = restart()
    patience_left = patience
    total_scroll = 0.0

    for step in range(int(1e5)):
        action = getAction(doodle, platforms, settings, weights)
        apply_action(doodle, platforms, settings, action)
        
        y_before = platforms[0]['y']

        if loop(doodle, platforms, settings) == -1:
            break

        y_after = platforms[0]['y']
        scroll = y_after - y_before

        if scroll > 0 and scroll < 100:
            total_scroll += scroll
            patience_left = patience
        else:
            patience_left -= 1

        if patience_left <= 0:
            break

    # ИСПРАВЛЕНО: Основной счет - это реальное количество платформ
    # Микро-бонус за скролл (0-0.99) добавлен для плавности градиента мутаций
    score = settings['score'] + (total_scroll % 100) / 100.0

    if patience_left > 0 and step >= int(1e5) - 10:
        score += 5000.0

    return score

def mutate(weights, mutation_rate=0.02, mutation_prob=0.1):
    W1, W2, W3 = weights

    def mutate_layer(layer):
        dW = np.random.normal(size=layer.shape)
        mask = np.random.random(size=layer.shape) < mutation_prob
        return layer + dW * mask * mutation_rate

    return mutate_layer(W1), mutate_layer(W2), mutate_layer(W3)

def crossover(w1, w2):
    W1_a, W2_a, W3_a = w1
    W1_b, W2_b, W3_b = w2

    def cross_matrices(m1, m2):
        mask = np.random.random(size=m1.shape) < 0.5
        return np.where(mask, m1, m2)

    return cross_matrices(W1_a, W1_b), cross_matrices(W2_a, W2_b), cross_matrices(W3_a, W3_b)

def generate_random(population, size, current_mut_rate=0.02):
    new_population = []
    for _ in range(size):
        if np.random.random() < 0.5:
            new_population.append(get_one())
        else:
            parent_idx = np.random.randint(len(population))
            new_population.append(mutate(population[parent_idx], mutation_rate=current_mut_rate))
    return new_population

def selection(population, scores, topK=4):
    scores = np.array(scores, dtype=float)
    if scores.sum() > 0:
        scores /= scores.sum()
    else:
        scores = np.ones_like(scores) / len(scores)

    elitismTopK = np.argsort(scores)[::-1][:topK]
    new_population = [
        tuple(layer.copy() for layer in population[i])
        for i in elitismTopK
    ]
    return new_population

def breed(population, scores, nChilds=10, current_mut_rate=0.02):
    scores = np.array(scores) * 1.0
    if scores.sum() > 0:
        scores /= scores.sum()
    else:
        scores = np.ones_like(scores) / len(scores)
        
    parents = np.random.choice(len(scores), p=scores, size=(nChilds, 2))

    new_population = []
    for parentA, parentB in parents:
        new_population.append(mutate(crossover(population[parentA], population[parentB]),
                                     mutation_rate=current_mut_rate))
    return new_population

def get_new_population(population, scores, topK=4, randomNum=10, current_mut_rate=0.02):
    return (
        selection(population, scores, topK) +
        breed(
            population,
            scores,
            nChilds=max(0, len(population) - randomNum - topK),
            current_mut_rate=current_mut_rate
        ) +
        generate_random(population, randomNum, current_mut_rate)
    )

def save_thingey(weights, score, prefix="doodlejump_avg"):
    W1, W2, W3 = weights
    filename = f'{prefix}_{score:.1f}.js'
    with open(filename, 'w') as f:
        f.write(f"const W1 = {json.dumps(W1.tolist())};\n")
        f.write(f"const W2 = {json.dumps(W2.tolist())};\n")
        f.write(f"const W3 = {json.dumps(W3.tolist())};\n")
    print(f"Файл {filename} успешно сохранен!")

POPULATION_SIZE = 128
RANDOM_SIZE = 16
ELITE_SIZE = 8
NUM_GENERATIONS = 150
NUM_REPEATS = 6
NUM_RESTARTS = 3

best_thingey = None
best_mean_score = 0

def evaluate_agent(args):
    weights, seed = args
    np.random.seed(seed)
    return get_score(weights, patience=300)

def main():
    global best_thingey, best_mean_score
    cores = multiprocessing.cpu_count()

    with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
        for n_restart in range(NUM_RESTARTS):
            print('=' * 50)
            print('Старт перезапуска №%d' % (n_restart + 1))
            print('=' * 50)

            population = [get_one() for _ in range(POPULATION_SIZE)]
            stagnation_counter = 0
            local_best_mean = 0

            for generation in range(NUM_GENERATIONS):
                agent_scores = np.zeros((POPULATION_SIZE, NUM_REPEATS))
                seeds = [np.random.randint(0, 2**31 - 1) for _ in range(NUM_REPEATS)]
                
                for i, seed in enumerate(seeds):
                    tasks = [(weights, seed) for weights in population]
                    results = list(tqdm(
                        executor.map(evaluate_agent, tasks),
                        total=POPULATION_SIZE,
                        desc=f'Поколение {generation}, Сид {i+1}/{NUM_REPEATS}',
                        leave=False
                    ))
                    agent_scores[:, i] = results

                mean_scores = np.mean(agent_scores, axis=1)
                min_scores = np.min(agent_scores, axis=1)
                max_scores = np.max(agent_scores, axis=1)

                fitness_scores = mean_scores + min_scores * 2.0
                dynamic_mut_rate = max(0.015, 0.15 * (0.985 ** generation))

                best_idx = np.argmax(fitness_scores)
                current_best_mean = mean_scores[best_idx]
                current_best_min = min_scores[best_idx]

                print(
                    f'Итоги {generation} | '
                    f'Лучший СРЕДНИЙ: {current_best_mean:.1f} | '
                    f'Мин. провал: {current_best_min:.1f} | '
                    f'Абсолют. макс: {np.max(max_scores):.1f} | '
                    f'Мутация: {dynamic_mut_rate:.3f}'
                )

                if current_best_mean > best_mean_score:
                    best_mean_score = current_best_mean
                    best_thingey = [layer.copy() for layer in population[best_idx]]
                    print(f'>>> НОВЫЙ РЕКОРД СРЕДНЕГО СЧЕТА! {best_mean_score:.1f} (Провал не ниже {current_best_min:.1f}) <<<')
                    if best_mean_score > 10.0:
                        save_thingey(best_thingey, best_mean_score, prefix="doodlejump_avg")

                if current_best_mean > local_best_mean:
                    local_best_mean = current_best_mean
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                selection_scores = fitness_scores ** 2 
                if selection_scores.sum() == 0:
                    selection_scores = np.ones_like(fitness_scores)

                population = get_new_population(
                    population,
                    selection_scores,
                    ELITE_SIZE,
                    RANDOM_SIZE,
                    dynamic_mut_rate
                )

                if stagnation_counter >= 35:
                    print('Эволюция застряла. Переходим к следующему рестарту!')
                    break

    if best_thingey is not None and best_mean_score > 10.0:
        save_thingey(best_thingey, best_mean_score, prefix="doodlejump_final_avg")

if __name__ == "__main__":
    main()