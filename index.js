
const ARRAY_TYPES = [
    Int8Array, Uint8Array, Uint8ClampedArray, Int16Array, Uint16Array,
    Int32Array, Uint32Array, Float32Array, Float64Array
];

/** @typedef {Int8ArrayConstructor | Uint8ArrayConstructor | Uint8ClampedArrayConstructor | Int16ArrayConstructor | Uint16ArrayConstructor | Int32ArrayConstructor | Uint32ArrayConstructor | Float32ArrayConstructor | Float64ArrayConstructor} TypedArrayConstructor */

const VERSION = 2; // serialized format version
const HEADER_SIZE = 16;

export default class KDBush {

    /**
     * Creates an index from raw `ArrayBuffer` data.
     * @param {ArrayBuffer} data
     */
    static from(data) {
        if (!(data instanceof ArrayBuffer)) {
            throw new Error('Data must be an instance of ArrayBuffer.');
        }
        const [magic, versionAndType] = new Uint8Array(data, 0, 2);
        if (magic !== 0xdb) {
            throw new Error('Data does not appear to be in a KDBush format.');
        }
        const version = versionAndType >> 4;
        if (version !== VERSION) {
            throw new Error(`Got v${version} data when expected v${VERSION}.`);
        }
        const ArrayType = ARRAY_TYPES[versionAndType & 0x0f];
        if (!ArrayType) {
            throw new Error('Unrecognized array type.');
        }
        const [nodeSize] = new Uint16Array(data, 2, 1);
        const [numItems] = new Uint32Array(data, 4, 1);
        const [dimensions] = new Uint8Array(data, 8, 1);

        return new KDBush(numItems, nodeSize, ArrayType, dimensions, data);
    }

    /**
     * Creates an index that will hold a given number of items.
     * @param {number} numItems
     * @param {number} [nodeSize=64] Size of the KD-tree node (64 by default).
     * @param {TypedArrayConstructor} [ArrayType=Float64Array] The array type used for coordinates storage (`Float64Array` by default).
     * @param {number} [dimensions=2] Number of dimensions for each point (2 by default).
     * @param {ArrayBuffer} [data] (For internal use only)
     */
    constructor(numItems, nodeSize = 64, ArrayType = Float64Array, dimensions = 2, data) {
        if (isNaN(numItems) || numItems < 0) throw new Error(`Unexpected numItems value: ${numItems}.`);
        if (dimensions < 2 || dimensions > 255) throw new Error(`dimensions must be between 2 and 255: ${dimensions}`);

        this.numItems = +numItems;
        this.nodeSize = Math.min(Math.max(+nodeSize, 2), 65535);
        this.dimensions = dimensions;
        this.ArrayType = ArrayType;
        this.IndexArrayType = numItems < 65536 ? Uint16Array : Uint32Array;

        const arrayTypeIndex = ARRAY_TYPES.indexOf(this.ArrayType);
        const coordsByteSize = numItems * dimensions * this.ArrayType.BYTES_PER_ELEMENT;
        const idsByteSize = numItems * this.IndexArrayType.BYTES_PER_ELEMENT;
        const padCoords = (8 - idsByteSize % 8) % 8;

        if (arrayTypeIndex < 0) {
            throw new Error(`Unexpected typed array class: ${ArrayType}.`);
        }

        if (data && (data instanceof ArrayBuffer)) { // reconstruct an index from a buffer
            this.data = data;
            this.ids = new this.IndexArrayType(this.data, HEADER_SIZE, numItems);
            this.coords = new this.ArrayType(this.data, HEADER_SIZE + idsByteSize + padCoords, numItems * dimensions);
            this._pos = numItems * dimensions;
            this._finished = true;
        } else { // initialize a new index
            this.data = new ArrayBuffer(HEADER_SIZE + coordsByteSize + idsByteSize + padCoords);
            this.ids = new this.IndexArrayType(this.data, HEADER_SIZE, numItems);
            this.coords = new this.ArrayType(this.data, HEADER_SIZE + idsByteSize + padCoords, numItems * dimensions);
            this._pos = 0;
            this._finished = false;

            // set header
            new Uint8Array(this.data, 0, 2).set([0xdb, (VERSION << 4) + arrayTypeIndex]);
            new Uint16Array(this.data, 2, 1)[0] = nodeSize;
            new Uint32Array(this.data, 4, 1)[0] = numItems;
            new Uint8Array(this.data, 8, 1)[0] = dimensions;
        }
    }

    /**
     * Add a point to the index.
     * @param {number} x
     * @param {number} y
     * @param {number[]} dims
     * @returns {number} An incremental index associated with the added item (starting from `0`).
     */
    add(x, y, ...dims) {
        if (dims.length !== this.dimensions - 2) throw new Error(`Expected ${this.dimensions} dimensions`);

        const index = this._pos >> 1;
        this.ids[index] = index;
        this.coords[this._pos++] = x;
        this.coords[this._pos++] = y;
        for (const dim of dims) {
            this.coords[this._pos++] = dim;
        }
        return index;
    }

    /**
     * Perform indexing of the added points.
     */
    finish() {
        const numAdded = this._pos / this.dimensions;
        if (numAdded !== this.numItems) {
            throw new Error(`Added ${numAdded} items when expected ${this.numItems}.`);
        }
        // kd-sort both arrays for efficient search
        sort(this.ids, this.coords, this.nodeSize, this.dimensions, 0, this.numItems - 1, 0);

        this._finished = true;
        return this;
    }

    /**
     * Search the index for items within a given bounding box.
     * @param {number[]} bounds Pairs of (min, max) for each dimension.
     * @returns {number[]} An array of indices correponding to the found items.
     */
    range(...bounds) {
        if (!this._finished) throw new Error('Data not yet indexed - call index.finish().');
        if (bounds.length !== this.dimensions * 2) throw new Error(`Expected ${this.dimensions} bounds`);

        const {ids, coords, nodeSize, dimensions} = this;
        const stack = [0, ids.length - 1, 0];
        const result = [];

        // recursively search for items in range in the kd-sorted arrays
        while (stack.length) {
            const axis = stack.pop() || 0;
            const right = stack.pop() || 0;
            const left = stack.pop() || 0;

            // if we reached "tree node", search linearly
            if (right - left <= nodeSize) {
                for (let i = left; i <= right; i++) {
                    const point = extract(coords, dimensions, i);
                    if (withinBounds(point, bounds)) result.push(ids[i]);
                }
                continue;
            }

            // otherwise find the middle index
            const m = (left + right) >> 1;

            // include the middle item if it's in range
            const point = extract(coords, dimensions, m);
            if (withinBounds(point, bounds)) result.push(ids[m]);

            // queue search in axes that intersect the query
            const min = bounds[axis * 2];
            const max = bounds[axis * 2 + 1];
            const coord = point[axis];
            if (min <= coord) {
                stack.push(left);
                stack.push(m - 1);
                stack.push((axis + 1) % dimensions);
            }
            if (max >= coord) {
                stack.push(m + 1);
                stack.push(right);
                stack.push((axis + 1) % dimensions);
            }
        }

        return result;
    }

    /**
     * Search the index for items within a given radius around a center.
     * @param {number[]} center Coordinates of the center of the query.
     * @param {number} r Query radius.
     * @returns {number[]} An array of indices correponding to the found items.
     */
    within(r, ...center) {
        if (!this._finished) throw new Error('Data not yet indexed - call index.finish().');

        const {ids, coords, nodeSize, dimensions} = this;
        const stack = [0, ids.length - 1, 0];
        const result = [];
        const r2 = r * r;

        // recursively search for items within radius in the kd-sorted arrays
        while (stack.length) {
            const axis = stack.pop() || 0;
            const right = stack.pop() || 0;
            const left = stack.pop() || 0;

            // if we reached "tree node", search linearly
            if (right - left <= nodeSize) {
                for (let i = left; i <= right; i++) {
                    const point = extract(coords, dimensions, i);
                    if (sqDist(point, center) <= r2) result.push(ids[i]);
                }
                continue;
            }

            // otherwise find the middle index
            const m = (left + right) >> 1;

            // include the middle item if it's in range
            const point = extract(coords, dimensions, m);
            if (sqDist(point, center) <= r2) result.push(ids[m]);

            // queue search in halves that intersect the query
            const coord = point[axis];
            const qcoord = center[axis];
            if (qcoord - r <= coord) {
                stack.push(left);
                stack.push(m - 1);
                stack.push((axis + 1) % dimensions);
            }
            if (qcoord + r >= coord) {
                stack.push(m + 1);
                stack.push(right);
                stack.push((axis + 1) % dimensions);
            }
        }

        return result;
    }
}

/**
 * @param {Uint16Array | Uint32Array} ids
 * @param {InstanceType<TypedArrayConstructor>} coords
 * @param {number} nodeSize
 * @param {number} dimensions
 * @param {number} left
 * @param {number} right
 * @param {number} axis
 */
function sort(ids, coords, nodeSize, dimensions, left, right, axis) {
    if (right - left <= nodeSize) return;

    const m = (left + right) >> 1; // middle index

    // sort ids and coords around the middle index so that the halves lie
    // either left/right or top/bottom correspondingly (taking turns)
    select(ids, coords, dimensions, m, left, right, axis);

    // recursively kd-sort first half and second half on the next axis
    sort(ids, coords, nodeSize, dimensions, left, m - 1, (axis + 1) % dimensions);
    sort(ids, coords, nodeSize, dimensions, m + 1, right, (axis + 1) % dimensions);
}

/**
 * Custom Floyd-Rivest selection algorithm: sort ids and coords so that
 * [left..k-1] items are smaller than k-th item (on either x or y axis)
 * @param {Uint16Array | Uint32Array} ids
 * @param {InstanceType<TypedArrayConstructor>} coords
 * @param {number} dimensions
 * @param {number} k
 * @param {number} left
 * @param {number} right
 * @param {number} axis
 */
function select(ids, coords, dimensions, k, left, right, axis) {

    while (right > left) {
        if (right - left > 600) {
            const n = right - left + 1;
            const m = k - left + 1;
            const z = Math.log(n);
            const s = 0.5 * Math.exp(2 * z / 3);
            const sd = 0.5 * Math.sqrt(z * s * (n - s) / n) * (m - n / 2 < 0 ? -1 : 1);
            const newLeft = Math.max(left, Math.floor(k - m * s / n + sd));
            const newRight = Math.min(right, Math.floor(k + (n - m) * s / n + sd));
            select(ids, coords, dimensions, k, newLeft, newRight, axis);
        }

        const t = coords[dimensions * k + axis];
        let i = left;
        let j = right;

        swapItem(ids, coords, dimensions, left, k);
        if (coords[dimensions * right + axis] > t) swapItem(ids, coords, dimensions, left, right);

        while (i < j) {
            swapItem(ids, coords, dimensions, i, j);
            i++;
            j--;
            while (coords[dimensions * i + axis] < t) i++;
            while (coords[dimensions * j + axis] > t) j--;
        }

        if (coords[dimensions * left + axis] === t) swapItem(ids, coords, dimensions, left, j);
        else {
            j++;
            swapItem(ids, coords, dimensions, j, right);
        }

        if (j <= k) left = j + 1;
        if (k <= j) right = j - 1;
    }
}

/**
 * @param {Uint16Array | Uint32Array} ids
 * @param {InstanceType<TypedArrayConstructor>} coords
 * @param {number} dimensions
 * @param {number} i
 * @param {number} j
 */
function swapItem(ids, coords, dimensions, i, j) {
    swap(ids, i, j);
    for (let d = 0; d < dimensions; d++) {
        swap(coords, dimensions * i + d, dimensions * j + d);
    }
}

/**
 * @param {InstanceType<TypedArrayConstructor>} arr
 * @param {number} i
 * @param {number} j
 */
function swap(arr, i, j) {
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
}

/**
 * @param {number[]} a
 * @param {number[]} b
 */
function sqDist(a, b) {
    let dist = 0;
    for (let i = 0; i < a.length; i++) {
        const d = a[i] - b[i];
        dist += d * d;
    }
    return dist;
}

/**
 * @param {number[]} coords
 * @param {number[]} ranges
 */
function withinBounds(coords, ranges) {
    for (let axis = 0; axis < coords.length; axis++) {
        const coord = coords[axis];
        const min = ranges[axis * 2 + 0];
        const max = ranges[axis * 2 + 1];
        if (coord < min || coord > max) return false;
    }
    return true;
}

/**
 * @param {InstanceType<TypedArrayConstructor>} coords
 * @param {number} dimensions
 * @param {number} item
 * @returns {number[]}
 * 
 */
function extract(coords, dimensions, item) {
    let point = Array(dimensions);
    for (let i = 0; i < point.length; i++) {
        point[i] = coords[dimensions * item + i];
    }
    return point;
}
